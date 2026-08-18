// crr tree on the c++ side: Hull's 5 step put, convergence to black scholes, the american call
// w/o dividends, and the errors
#include <gtest/gtest.h>

#include <stdexcept>

#include "ope/black_scholes.hpp"
#include "ope/tree.hpp"

using ope::Option;

TEST(Tree, HullFiveStepAmericanPut) {
    // S=50, K=50, T=5 months, r=10%, sigma=40%, 5 steps. Hull rounds it to 4.49
    EXPECT_NEAR(ope::crr_price(Option::put, 50, 50, 5.0 / 12, 0.10, 0.40, 0, 5, true),
                4.488458534726, 1e-11);
}

TEST(Tree, EuropeanConvergesToBlackScholes) {
    for (Option o : {Option::call, Option::put}) {
        const double exact = ope::bs_price(o, 100, 110, 1, 0.05, 0.25, 0.02);
        // averaging n and n + 1 steps cancels most of the odd/even zigzag
        const double tree = 0.5 * (ope::crr_price(o, 100, 110, 1, 0.05, 0.25, 0.02, 2000, false) +
                                   ope::crr_price(o, 100, 110, 1, 0.05, 0.25, 0.02, 2001, false));
        EXPECT_NEAR(tree, exact, 1e-4);
    }
}

TEST(Tree, AmericanCallWithoutDividendsIsTheEuropeanCall) {
    EXPECT_EQ(ope::crr_price(Option::call, 42, 40, 0.5, 0.10, 0.20, 0, 500, true),
              ope::crr_price(Option::call, 42, 40, 0.5, 0.10, 0.20, 0, 500, false));
}

TEST(Tree, DividendsMakeEarlyExerciseOfACallWorthSomething) {
    const double american = ope::crr_price(Option::call, 100, 80, 1, 0.03, 0.25, 0.08, 1000, true);
    EXPECT_GT(american - ope::bs_price(Option::call, 100, 80, 1, 0.03, 0.25, 0.08), 2.0);
}

TEST(Tree, BadInputsThrow) {
    EXPECT_THROW(ope::crr_price(Option::call, -1, 40, 1, 0.05, 0.2, 0, 10, false), std::invalid_argument);
    EXPECT_THROW(ope::crr_price(Option::call, 42, 40, 1, 0.05, 0.2, 0, 0, false), std::invalid_argument);
    EXPECT_THROW(ope::crr_price(Option::call, 42, 40, 1, 0.05, 0.0, 0, 10, false), std::invalid_argument);
    // e^{0.5} is above u = e^{0.01}, so p > 1
    EXPECT_THROW(ope::crr_price(Option::call, 42, 40, 1, 0.5, 0.01, 0, 1, false), std::invalid_argument);
}
