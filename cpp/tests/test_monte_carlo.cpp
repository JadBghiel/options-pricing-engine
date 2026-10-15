// monte carlo on the c++ side: the generator against published answers, the normals'
// moments, and the estimator against black scholes, same seed must give the same bits for
// any thread count
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "ope/black_scholes.hpp"
#include "ope/monte_carlo.hpp"
#include "ope/philox.hpp"

using ope::Option;

TEST(Philox, MatchesRandom123KnownAnswers) {
    // known answer vectors of philox4x64_10 from the Random123 distribution
    EXPECT_EQ(ope::philox4x64({0, 0, 0, 0}, {0, 0}),
              (ope::PhiloxBlock{0x16554d9eca36314cULL, 0xdb20fe9d672d0fdcULL,
                                0xd7e772cee186176bULL, 0x7e68b68aec7ba23bULL}));
    constexpr std::uint64_t ff = ~0ULL;
    EXPECT_EQ(ope::philox4x64({ff, ff, ff, ff}, {ff, ff}),
              (ope::PhiloxBlock{0x87b092c3013fe90bULL, 0x438c3c67be8d0224ULL,
                                0x9cc7d7c69cd777b6ULL, 0xa09caebf594f0ba0ULL}));
    EXPECT_EQ(ope::philox4x64({0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
                               0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL},
                              {0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL}),
              (ope::PhiloxBlock{0xa528f45403e61d95ULL, 0x38c72dbd566e9788ULL,
                                0xa5a1610e72fd18b5ULL, 0x57bd43b5e52b7fe6ULL}));
}

TEST(Normals, HaveMeanZeroVarianceOneAndNoExcessKurtosis) {
    const std::size_t n = 1'000'000;
    std::vector<double> z(n);
    ope::fill_normals(z, 42);
    double m1 = 0, m2 = 0, m4 = 0;
    for (double v : z) {
        m1 += v;
        m2 += v * v;
        m4 += v * v * v * v;
    }
    m1 /= n;
    m2 /= n;
    m4 /= n;
    // standard errors at 1e6 draws: mean 0.001, variance 0.0014, 4th moment ~0.01
    EXPECT_NEAR(m1, 0.0, 0.005);
    EXPECT_NEAR(m2, 1.0, 0.007);
    EXPECT_NEAR(m4, 3.0, 0.05);
}

TEST(Normals, AnyStartingPointGivesTheSameStream) {
    std::vector<double> all(100), tail(37);
    ope::fill_normals(all, 7);
    ope::fill_normals(tail, 7, 63);  // starts mid block on purpose
    for (std::size_t i = 0; i < tail.size(); ++i) {
        EXPECT_EQ(tail[i], all[63 + i]);
    }
}

TEST(MonteCarlo, SameSeedSameBitsForAnyThreadCount) {
    const auto one = ope::mc_price(Option::call, 42, 40, 0.5, 0.1, 0.2, 0, 300'000, true, true, 5, 1);
    for (int threads : {2, 3, 8}) {
        const auto many =
            ope::mc_price(Option::call, 42, 40, 0.5, 0.1, 0.2, 0, 300'000, true, true, 5, threads);
        EXPECT_EQ(one.estimate, many.estimate);
        EXPECT_EQ(one.std_error, many.std_error);
    }
}

TEST(MonteCarlo, SeedPathEqualsTheGivenNormalsPath) {
    const std::int64_t n = 100'003;  // not a multiple of the chunk or of 4
    std::vector<double> z(n);
    ope::fill_normals(z, 9);
    const auto seeded = ope::mc_price(Option::put, 42, 40, 0.5, 0.1, 0.2, 0.01, n, false, true, 9, 4);
    const auto given = ope::mc_price_from_normals(Option::put, 42, 40, 0.5, 0.1, 0.2, 0.01, z, false, true);
    EXPECT_EQ(seeded.estimate, given.estimate);
    EXPECT_EQ(seeded.std_error, given.std_error);
    EXPECT_EQ(seeded.n_paths, n);
}

TEST(MonteCarlo, WithinFourStandardErrorsOfBlackScholes) {
    for (Option o : {Option::call, Option::put}) {
        const double exact = ope::bs_price(o, 42, 40, 0.5, 0.1, 0.2);
        for (bool anti : {false, true}) {
            for (bool control : {false, true}) {
                const auto res = ope::mc_price(o, 42, 40, 0.5, 0.1, 0.2, 0, 200'000, anti, control, 11);
                EXPECT_LT(std::abs(res.estimate - exact), 4 * res.std_error);
            }
        }
    }
}

TEST(MonteCarlo, ZeroVolIsExact) {
    const auto res = ope::mc_price(Option::call, 100, 90, 1, 0.05, 0.0, 0, 10, false, false, 0);
    EXPECT_NEAR(res.estimate, 100 - 90 * std::exp(-0.05), 1e-12);
    EXPECT_LT(res.std_error, 1e-12);
}

TEST(MonteCarlo, BadInputsThrow) {
    auto price = [](double S, double T, double sigma, std::int64_t n, bool anti) {
        return ope::mc_price(Option::call, S, 40, T, 0.1, sigma, 0, n, anti, false, 0);
    };
    EXPECT_THROW(price(-1, 0.5, 0.2, 100, false), std::invalid_argument);
    EXPECT_THROW(price(42, 0, 0.2, 100, false), std::invalid_argument);
    EXPECT_THROW(price(42, 0.5, -0.1, 100, false), std::invalid_argument);
    EXPECT_THROW(price(NAN, 0.5, 0.2, 100, false), std::invalid_argument);
    EXPECT_THROW(price(42, 0.5, 0.2, 1, false), std::invalid_argument);
    EXPECT_THROW(price(42, 0.5, 0.2, 101, true), std::invalid_argument);
    EXPECT_THROW(price(42, 0.5, 0.2, 2, true), std::invalid_argument);  // 1 pair, no spread
    EXPECT_THROW(ope::mc_price(Option::call, 42, 40, 0.5, 0.1, 0.2, 0, 100, false, false, 0, -1),
                 std::invalid_argument);
}
