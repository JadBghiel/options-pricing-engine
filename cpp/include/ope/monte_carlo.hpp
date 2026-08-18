// monte carlo for european options under black scholes, same estimator as src/ope/monte_carlo.py
//
// S_T is simulated exactly in 1 step. the draws come from philox w/ the seed as key, normal
// number i always comes from the same counter, so a seed gives the same price for any number
// of threads
#pragma once

#include <cstdint>
#include <span>

#include "ope/option.hpp"

namespace ope {

struct MCResult {
    double estimate;
    double std_error;
    std::int64_t n_paths;
};

// paths are cut into chunks of this many base draws. each chunk is summed on its own and the
// chunks are merged in order, so the result never depends on which thread did which chunk
inline constexpr std::int64_t mc_chunk = 1 << 14;

// draws n_paths paths w/ normals from philox(seed), on `threads` threads (0 = all cores)
MCResult mc_price(Option option, double S, double K, double T, double r, double sigma, double q,
                  std::int64_t n_paths, bool antithetic, bool control_variate, std::uint64_t seed,
                  int threads = 0);

// same estimator on given normals: z.size() paths, or 2 * z.size() w/ antithetic (z and -z).
// python and c++ fed the same z must agree to rounding
MCResult mc_price_from_normals(Option option, double S, double K, double T, double r, double sigma,
                               double q, std::span<const double> z, bool antithetic,
                               bool control_variate);

// standard normals number first, first + 1, ... of the stream of `seed`
void fill_normals(std::span<double> out, std::uint64_t seed, std::uint64_t first = 0);

}  // namespace ope
