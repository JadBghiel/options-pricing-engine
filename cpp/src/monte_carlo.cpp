#include "ope/monte_carlo.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ope/parallel.hpp"
#include "ope/philox.hpp"

namespace ope {

namespace {

// box muller on the 4 words of 1 philox block -> 4 independent standard normals
std::array<double, 4> block_normals(const PhiloxBlock& b) {
    std::array<double, 4> z{};
    for (std::size_t pair = 0; pair < 2; ++pair) {
        // top 53 bits -> a double. u1 in (0, 1] so log(u1) stays finite, u2 in [0, 1)
        const double u1 = static_cast<double>((b[2 * pair] >> 11) + 1) * 0x1.0p-53;
        const double u2 = static_cast<double>(b[2 * pair + 1] >> 11) * 0x1.0p-53;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * std::numbers::pi * u2;
        z[2 * pair] = radius * std::cos(angle);
        z[2 * pair + 1] = radius * std::sin(angle);
    }
    return z;
}

// what every path needs, computed once
struct Setup {
    Option option;
    double S;
    double K;
    double disc;    // e^{-rT}
    double drift;   // (r - q - sigma^2/2) T
    double vol;     // sigma sqrt(T)
    double target;  // S e^{-qT}, the known mean of the control
};

Setup make_setup(Option option, double S, double K, double T, double r, double sigma, double q) {
    return {option,
            S,
            K,
            std::exp(-r * T),
            (r - q - 0.5 * (sigma * sigma)) * T,
            sigma * std::sqrt(T),
            S * std::exp(-q * T)};
}

// count, means and centered sums of x (discounted payoff) and y (discounted stock)
struct Moments {
    std::int64_t n = 0;
    double mx = 0.0;
    double my = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
};

// chan, golub and leveque: 2 groups' moments -> the moments of the 2 together, exact algebra
Moments merge(const Moments& a, const Moments& b) {
    if (a.n == 0) {
        return b;
    }
    const auto na = static_cast<double>(a.n);
    const auto nb = static_cast<double>(b.n);
    const double n = na + nb;
    const double dx = b.mx - a.mx;
    const double dy = b.my - a.my;
    const double w = na * nb / n;
    return {a.n + b.n,         a.mx + dx * nb / n,     a.my + dy * nb / n,
            a.sxx + b.sxx + dx * dx * w, a.syy + b.syy + dy * dy * w, a.sxy + b.sxy + dx * dy * w};
}

// 1 chunk of base draws -> its moments, 2 passes (means, then centered sums), stable in floats
Moments chunk_moments(const Setup& s, std::span<const double> z, bool antithetic,
                      std::vector<double>& x, std::vector<double>& y) {
    const std::size_t n = z.size();
    double sx = 0.0;
    double sy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double up = s.S * std::exp(s.drift + s.vol * z[i]);
        double xi = s.disc * payoff(s.option, up, s.K);
        double yi = s.disc * up;
        if (antithetic) {
            // the mirror path -z, averaged w/ its twin: the pair is 1 independent sample
            const double down = s.S * std::exp(s.drift + s.vol * -z[i]);
            xi = 0.5 * (xi + s.disc * payoff(s.option, down, s.K));
            yi = 0.5 * (yi + s.disc * down);
        }
        x[i] = xi;
        y[i] = yi;
        sx += xi;
        sy += yi;
    }
    Moments m;
    m.n = static_cast<std::int64_t>(n);
    m.mx = sx / static_cast<double>(n);
    m.my = sy / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - m.mx;
        const double dy = y[i] - m.my;
        m.sxx += dx * dx;
        m.syy += dy * dy;
        m.sxy += dx * dy;
    }
    return m;
}

// estimate and standard error from the merged moments, w/ or w/o the control variate
MCResult finish(const Moments& t, bool control_variate, double target, std::int64_t n_paths) {
    double mean = t.mx;
    double ss = t.sxx;
    if (control_variate) {
        // regression slope of payoff on stock, the beta that cuts the variance the most
        const double beta = t.syy > 0.0 ? t.sxy / t.syy : 0.0;
        mean = t.mx - beta * (t.my - target);
        ss = t.sxx - 2.0 * beta * t.sxy + beta * beta * t.syy;
    }
    const auto m = static_cast<double>(t.n);
    const double var = std::max(ss, 0.0) / (m - 1.0);  // rounding can push ss a hair below 0
    return {mean, std::sqrt(var) / std::sqrt(m), n_paths};
}

// runs every chunk, on up to `threads` threads. fill(begin, buffer) returns the base draws
// begin, begin + 1, ... of 1 chunk
template <class Fill>
MCResult run(const Setup& s, std::int64_t m, std::int64_t n_paths, bool antithetic,
             bool control_variate, int threads, Fill fill) {
    const std::int64_t n_chunks = (m + mc_chunk - 1) / mc_chunk;
    std::vector<Moments> parts(static_cast<std::size_t>(n_chunks));
    std::atomic<std::int64_t> next{0};

    auto worker = [&] {
        std::vector<double> z(mc_chunk);
        std::vector<double> x(mc_chunk);
        std::vector<double> y(mc_chunk);
        // each thread grabs the next free chunk until none is left -> fast threads do more
        for (std::int64_t c = next++; c < n_chunks; c = next++) {
            const std::int64_t begin = c * mc_chunk;
            const auto count = static_cast<std::size_t>(std::min(mc_chunk, m - begin));
            const std::span<const double> draws = fill(begin, std::span<double>(z).first(count));
            parts[static_cast<std::size_t>(c)] = chunk_moments(s, draws, antithetic, x, y);
        }
    };

    const auto workers =
        static_cast<std::int64_t>(std::min<std::int64_t>(resolve_threads(threads), n_chunks));
    if (workers <= 1) {
        worker();
    } else {
        std::vector<std::jthread> pool;
        pool.reserve(static_cast<std::size_t>(workers));
        for (std::int64_t w = 0; w < workers; ++w) {
            pool.emplace_back(worker);
        }
    }  // joined here

    // LEARNING NOTE: floating point addition isnt associative, (a + b) + c can differ from
    // a + (b + c) in the last bit, if each thread summed whatever paths it happened to grab,
    // the price would change w/ the thread count and the timing -> the paths are cut into
    // fixed chunks, each chunk is summed alone, and the chunks are merged in chunk order:
    // same seed, same bits, whether 1 thread or 8 did the work
    Moments total;
    for (const Moments& part : parts) {
        total = merge(total, part);
    }
    return finish(total, control_variate, s.target, n_paths);
}

void check_inputs(double S, double K, double T, double sigma, std::int64_t n_paths) {
    if (!(std::isfinite(S) && std::isfinite(K) && std::isfinite(T) && std::isfinite(sigma))) {
        throw std::invalid_argument("inputs must be finite");
    }
    if (S <= 0 || K <= 0) {
        throw std::invalid_argument("spot S and strike K must be > 0");
    }
    if (T <= 0) {
        throw std::invalid_argument(
            "maturity T must be > 0, at expiry there is nothing to simulate");
    }
    if (sigma < 0) {
        throw std::invalid_argument("volatility sigma must be >= 0");
    }
    if (n_paths < 2) {
        throw std::invalid_argument("n_paths must be >= 2 to estimate a standard error");
    }
}

void check_samples(std::int64_t m) {
    if (m < 2) {
        throw std::invalid_argument(
            "need >= 2 independent draws to estimate a standard error (an antithetic pair "
            "counts as 1)");
    }
}

}  // namespace

void fill_normals(std::span<double> out, std::uint64_t seed, std::uint64_t first) {
    // LEARNING NOTE: normal number i comes from philox block i / 4, lane i % 4, under the key
    // seed, no generator state is passed around: a thread that starts at path 1,000,000
    // computes block 250,000, that is what makes the streams per thread independent and
    // reproducible, and what lets python ask for the normals the c++ side used
    std::size_t k = 0;
    std::uint64_t i = first;
    while (k < out.size()) {
        const auto z = block_normals(philox4x64({i / 4, 0, 0, 0}, {seed, 0}));
        for (std::uint64_t lane = i % 4; lane < 4 && k < out.size(); ++lane, ++k, ++i) {
            out[k] = z[lane];
        }
    }
}

MCResult mc_price(Option option, double S, double K, double T, double r, double sigma, double q,
                  std::int64_t n_paths, bool antithetic, bool control_variate, std::uint64_t seed,
                  int threads) {
    check_inputs(S, K, T, sigma, n_paths);
    if (antithetic && n_paths % 2 != 0) {
        throw std::invalid_argument(
            "antithetic sampling needs an even n_paths (paths come in pairs)");
    }
    const std::int64_t m = antithetic ? n_paths / 2 : n_paths;
    check_samples(m);
    resolve_threads(threads);  // throws on a negative count before any work
    auto fill = [seed](std::int64_t begin, std::span<double> buffer) {
        fill_normals(buffer, seed, static_cast<std::uint64_t>(begin));
        return std::span<const double>(buffer);
    };
    return run(make_setup(option, S, K, T, r, sigma, q), m, n_paths, antithetic, control_variate,
               threads, fill);
}

MCResult mc_price_from_normals(Option option, double S, double K, double T, double r, double sigma,
                               double q, std::span<const double> z, bool antithetic,
                               bool control_variate) {
    const auto m = static_cast<std::int64_t>(z.size());
    const std::int64_t n_paths = antithetic ? 2 * m : m;
    check_inputs(S, K, T, sigma, n_paths);
    check_samples(m);
    auto fill = [z](std::int64_t begin, std::span<double> buffer) {
        return z.subspan(static_cast<std::size_t>(begin), buffer.size());
    };
    return run(make_setup(option, S, K, T, r, sigma, q), m, n_paths, antithetic, control_variate,
               1, fill);
}

}  // namespace ope
