#include "ope/black_scholes.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "ope/parallel.hpp"

namespace ope {

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();

// everything the formulas share, computed once per option like _prepare in python
struct Market {
    double df_r;  // e^{-rT}, discounts the strike
    double df_q;  // e^{-qT}, strips the dividends paid before T
    double sqrt_T;
    bool degenerate;  // sigma*sqrt(T) = 0 or S = 0, no randomness left
    double d1;
    double d2;
};

void check(double S, double K, double T, double r, double sigma, double q) {
    if (!(std::isfinite(S) && std::isfinite(K) && std::isfinite(T) && std::isfinite(r) &&
          std::isfinite(sigma) && std::isfinite(q))) {
        throw std::invalid_argument("inputs must be finite");
    }
    if (S < 0) {
        throw std::invalid_argument("spot S must be >= 0");
    }
    if (K <= 0) {
        throw std::invalid_argument("strike K must be > 0");
    }
    if (T < 0) {
        throw std::invalid_argument("maturity T must be >= 0");
    }
    if (sigma < 0) {
        throw std::invalid_argument("volatility sigma must be >= 0");
    }
}

// assumes check() passed, same operation order as python -> the rounding mostly matches too
Market prepare(double S, double K, double T, double r, double sigma, double q) {
    Market m{};
    m.df_r = std::exp(-r * T);
    m.df_q = std::exp(-q * T);
    m.sqrt_T = std::sqrt(T);
    const double vol_sqrt_T = sigma * m.sqrt_T;
    m.degenerate = vol_sqrt_T == 0.0 || S == 0.0;
    if (m.degenerate) {
        // no randomness left: d1, d2 -> +-inf, N(d) is a 0/1 indicator on F > K, see the
        // learning note in black_scholes.py
        m.d1 = m.d2 = S * m.df_q > K * m.df_r ? inf : -inf;
    } else {
        m.d1 = (std::log(S / K) + (r - q + 0.5 * (sigma * sigma)) * T) / vol_sqrt_T;
        m.d2 = m.d1 - vol_sqrt_T;
    }
    return m;
}

double call_value(const Market& m, double S, double K) {
    return S * m.df_q * norm_cdf(m.d1) - K * m.df_r * norm_cdf(m.d2);
}

double put_value(const Market& m, double S, double K) {
    return K * m.df_r * norm_cdf(-m.d2) - S * m.df_q * norm_cdf(-m.d1);
}

Greeks greeks_of(Option option, const Market& m, double S, double K, double T, double r,
                 double sigma, double q) {
    const double n_d1 = norm_pdf(m.d1);
    Greeks g{};
    // gamma and vega are the same for calls and puts, parity is linear in S and has no sigma
    g.gamma = m.degenerate ? 0.0 : m.df_q * n_d1 / (S * sigma * m.sqrt_T);
    g.vega = S * m.df_q * n_d1 * m.sqrt_T;
    const double decay = m.degenerate ? 0.0 : -(S * m.df_q * n_d1 * sigma / (2.0 * m.sqrt_T));
    if (option == Option::call) {
        g.delta = m.df_q * norm_cdf(m.d1);
        g.theta = decay + q * S * m.df_q * norm_cdf(m.d1) - r * K * m.df_r * norm_cdf(m.d2);
        g.rho = K * T * m.df_r * norm_cdf(m.d2);
    } else {
        g.delta = -m.df_q * norm_cdf(-m.d1);
        g.theta = decay - q * S * m.df_q * norm_cdf(-m.d1) + r * K * m.df_r * norm_cdf(-m.d2);
        g.rho = -K * T * m.df_r * norm_cdf(-m.d2);
    }
    return g;
}

void check_chain(const Chain& c, std::size_t out_size) {
    const std::size_t n = c.S.size();
    if (c.K.size() != n || c.T.size() != n || c.r.size() != n || c.sigma.size() != n ||
        c.q.size() != n || out_size != n) {
        throw std::invalid_argument("every input and output of a chain must have the same length");
    }
    for (std::size_t i = 0; i < n; ++i) {
        check(c.S[i], c.K[i], c.T[i], c.r[i], c.sigma[i], c.q[i]);
    }
}

// below this many options per thread, starting a thread costs more than it saves
constexpr std::size_t min_options_per_thread = 4096;

}  // namespace

double norm_cdf(double x) {
    // erfc instead of 1 + erf: far in the left tail erf(x) is -1 + tiny -> 1 + erf
    // cancels to 0 and loses every digit, while erfc(-x) keeps them, same reason python uses
    // scipy ndtr and N(-d2) for the put
    return 0.5 * std::erfc(-x * std::numbers::sqrt2 / 2.0);
}

double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * std::numbers::pi);
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

std::size_t Chain::size() const {
    return S.size();
}

double bs_price(Option option, double S, double K, double T, double r, double sigma, double q) {
    check(S, K, T, r, sigma, q);
    const Market m = prepare(S, K, T, r, sigma, q);
    return option == Option::call ? call_value(m, S, K) : put_value(m, S, K);
}

Greeks bs_greeks(Option option, double S, double K, double T, double r, double sigma, double q) {
    check(S, K, T, r, sigma, q);
    return greeks_of(option, prepare(S, K, T, r, sigma, q), S, K, T, r, sigma, q);
}

void bs_price_chain(Option option, const Chain& c, std::span<double> out, int threads) {
    check_chain(c, out.size());
    parallel_for(c.size(), threads, min_options_per_thread, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const Market m = prepare(c.S[i], c.K[i], c.T[i], c.r[i], c.sigma[i], c.q[i]);
            out[i] = option == Option::call ? call_value(m, c.S[i], c.K[i])
                                            : put_value(m, c.S[i], c.K[i]);
        }
    });
}

void bs_greeks_chain(Option option, const Chain& c, const GreeksOut& out, int threads) {
    check_chain(c, out.delta.size());
    if (out.gamma.size() != c.size() || out.vega.size() != c.size() ||
        out.theta.size() != c.size() || out.rho.size() != c.size()) {
        throw std::invalid_argument("every input and output of a chain must have the same length");
    }
    parallel_for(c.size(), threads, min_options_per_thread, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const Market m = prepare(c.S[i], c.K[i], c.T[i], c.r[i], c.sigma[i], c.q[i]);
            const Greeks g = greeks_of(option, m, c.S[i], c.K[i], c.T[i], c.r[i], c.sigma[i], c.q[i]);
            out.delta[i] = g.delta;
            out.gamma[i] = g.gamma;
            out.vega[i] = g.vega;
            out.theta[i] = g.theta;
            out.rho[i] = g.rho;
        }
    });
}

}  // namespace ope
