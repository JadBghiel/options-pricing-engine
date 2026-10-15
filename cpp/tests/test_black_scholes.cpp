// black scholes on the c++ side: reference values recomputed by an independent python script
// (plain math + scipy.stats), plus parity, finite differences and the degenerate inputs
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "ope/black_scholes.hpp"

using ope::Option;

TEST(BlackScholes, hullCallAndPut) {
    EXPECT_NEAR(ope::bs_price(Option::call, 42, 40, 0.5, 0.10, 0.20), 4.7594223929, 1e-9);
    EXPECT_NEAR(ope::bs_price(Option::put, 42, 40, 0.5, 0.10, 0.20), 0.8085993729, 1e-9);
}

TEST(BlackScholes, hullGreeks) {
    const ope::Greeks g = ope::bs_greeks(Option::call, 49, 50, 20.0 / 52, 0.05, 0.20);
    EXPECT_NEAR(g.delta, 0.5216046611, 1e-9);
    EXPECT_NEAR(g.gamma, 0.0655440393, 1e-9);
    EXPECT_NEAR(g.vega, 12.1054798826, 1e-8);
    EXPECT_NEAR(g.theta, -4.3053298229, 1e-8);
    EXPECT_NEAR(g.rho, 8.9069619496, 1e-8);
}

TEST(BlackScholes, PutCallParityOnRandomSets) {
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> spot(50, 150), mat(0.1, 2.0), rate(-0.01, 0.10),
        vol(0.1, 0.6), div(0.0, 0.05);
    for (int i = 0; i < 500; ++i) {
        const double S = spot(gen), K = spot(gen), T = mat(gen), r = rate(gen), s = vol(gen),
                     q = div(gen);
        const double lhs = ope::bs_price(Option::call, S, K, T, r, s, q) -
                           ope::bs_price(Option::put, S, K, T, r, s, q);
        EXPECT_NEAR(lhs, S * std::exp(-q * T) - K * std::exp(-r * T), 1e-10);
    }
}

TEST(BlackScholes, GreeksMatchCentralDifferences) {
    const double S = 100, K = 95, T = 0.75, r = 0.03, s = 0.25, q = 0.02, h = 1e-4 * S;
    for (Option o : {Option::call, Option::put}) {
        const ope::Greeks g = ope::bs_greeks(o, S, K, T, r, s, q);
        const double up = ope::bs_price(o, S + h, K, T, r, s, q);
        const double mid = ope::bs_price(o, S, K, T, r, s, q);
        const double down = ope::bs_price(o, S - h, K, T, r, s, q);
        EXPECT_NEAR(g.delta, (up - down) / (2 * h), 1e-8);
        EXPECT_NEAR(g.gamma, (up - 2 * mid + down) / (h * h), 1e-6);
        const double hs = 1e-5;
        EXPECT_NEAR(g.vega,
                    (ope::bs_price(o, S, K, T, r, s + hs, q) - ope::bs_price(o, S, K, T, r, s - hs, q)) /
                        (2 * hs),
                    1e-6);
    }
}

TEST(BlackScholes, DegenerateInputsGiveIntrinsicValues) {
    // at expiry: intrinsic, zero vol: discounted intrinsic on the forward, zero spot: 0 / K e^{-rT}
    EXPECT_DOUBLE_EQ(ope::bs_price(Option::call, 50, 40, 0, 0.05, 0.2), 10.0);
    EXPECT_DOUBLE_EQ(ope::bs_price(Option::put, 50, 40, 0, 0.05, 0.2), 0.0);
    EXPECT_NEAR(ope::bs_price(Option::call, 100, 100, 1, 0.05, 0, 0.02),
                100 * std::exp(-0.02) - 100 * std::exp(-0.05), 1e-12);
    EXPECT_EQ(ope::bs_price(Option::call, 0, 100, 1, 0.05, 0.2), 0.0);
    EXPECT_NEAR(ope::bs_price(Option::put, 0, 100, 1, 0.05, 0.2), 100 * std::exp(-0.05), 1e-12);

    const ope::Greeks g = ope::bs_greeks(Option::call, 110, 100, 0, 0.05, 0.2);
    EXPECT_EQ(g.gamma, 0.0);
    EXPECT_EQ(g.vega, 0.0);
    EXPECT_EQ(g.delta, 1.0);
    EXPECT_TRUE(std::isfinite(g.theta));
}

TEST(BlackScholes, ChainMatchesScalarCallsForAnyThreadCount) {
    const std::size_t n = 20'000;  // above the threshold -> several threads run
    std::vector<double> S(n), K(n), T(n, 0.5), r(n, 0.03), s(n, 0.25), q(n, 0.01), out(n);
    for (std::size_t i = 0; i < n; ++i) {
        S[i] = 100.0;
        K[i] = 50.0 + 100.0 * static_cast<double>(i) / static_cast<double>(n);
    }
    const ope::Chain c{S, K, T, r, s, q};
    for (int threads : {1, 3, 8}) {
        ope::bs_price_chain(Option::put, c, out, threads);
        for (std::size_t i = 0; i < n; i += 997) {
            EXPECT_EQ(out[i], ope::bs_price(Option::put, S[i], K[i], T[i], r[i], s[i], q[i]));
        }
    }
}

TEST(BlackScholes, BadInputsThrow) {
    EXPECT_THROW(ope::bs_price(Option::call, -1, 100, 1, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(ope::bs_price(Option::call, 100, 0, 1, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(ope::bs_price(Option::call, 100, 100, -1, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(ope::bs_price(Option::call, 100, 100, 1, 0.05, -0.2), std::invalid_argument);
    EXPECT_THROW(ope::bs_price(Option::call, 100, 100, 1, NAN, 0.2), std::invalid_argument);
    EXPECT_THROW(ope::parse_option("straddle"), std::invalid_argument);

    std::vector<double> a(3, 1.0), b(2, 1.0), out(3);
    EXPECT_THROW(ope::bs_price_chain(Option::call, {a, b, a, a, a, a}, out), std::invalid_argument);
}
