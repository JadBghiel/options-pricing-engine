// option type, payoff and the input checks shared by the 3 pricers
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

// LEARNING NOTE: -ffast-math lets the compiler assume no inf and no nan, black scholes sets d1
// and d2 to +-inf on purpose when sigma*sqrt(T) = 0 -> under fast math the degenerate cases
// silently break, and the rounding stops matching python, we refuse to compile w/ it
#ifdef __FAST_MATH__
#error "ope needs IEEE floating point, never build it w/ -ffast-math"
#endif

namespace ope {

enum class Option { call, put };

// "call" / "put" -> Option, same error msg as the python side
inline Option parse_option(std::string_view name) {
    if (name == "call") {
        return Option::call;
    }
    if (name == "put") {
        return Option::put;
    }
    throw std::invalid_argument("option must be 'call' or 'put', got '" + std::string(name) + "'");
}

// value at expiry, max(S_T - K, 0) for a call and max(K - S_T, 0) for a put
inline double payoff(Option option, double s_t, double K) {
    return option == Option::call ? std::max(s_t - K, 0.0) : std::max(K - s_t, 0.0);
}

}  // namespace ope
