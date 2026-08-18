#include "ope/tree.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ope {

double crr_price(Option option, double S, double K, double T, double r, double sigma, double q,
                 int steps, bool american) {
    if (S <= 0 || K <= 0) {
        throw std::invalid_argument("spot S and strike K must be > 0");
    }
    if (steps < 1) {
        throw std::invalid_argument("steps must be >= 1");
    }
    if (T <= 0 || sigma <= 0) {
        throw std::invalid_argument("the tree needs T > 0 and sigma > 0");
    }
    const double dt = T / steps;
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double p = (std::exp((r - q) * dt) - d) / (u - d);
    if (!(0.0 < p && p < 1.0)) {
        char msg[200];
        std::snprintf(msg, sizeof msg,
                      "risk neutral probability p = %.4f is outside (0, 1): the steps are too long "
                      "for this rate and volatility, use more steps",
                      p);
        throw std::invalid_argument(msg);
    }
    const double disc = std::exp(-r * dt);

    // node j at step i has the stock at S u^(2j - i), so 2n + 1 distinct prices for the whole
    // tree: spot[k + n] = S u^k, computed once instead of once per node
    const auto n = static_cast<std::size_t>(steps);
    std::vector<double> spot(2 * n + 1);
    for (std::size_t k = 0; k <= 2 * n; ++k) {
        spot[k] = S * std::pow(u, static_cast<double>(k) - static_cast<double>(n));
    }

    std::vector<double> v(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        v[j] = payoff(option, spot[2 * j], K);
    }
    // LEARNING NOTE: backward induction in place on 1 array. node j at step i needs nodes j and
    // j + 1 of step i + 1, and going j = 0, 1, 2... overwrites v[j] only after its last use, so
    // the tree costs n + 1 doubles of memory, not n^2 / 2. python builds a new array per step
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double cont = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            // american: the better of waiting and exercising now, spot index 2j - i + n
            v[j] = american ? std::max(cont, payoff(option, spot[2 * j + n - i], K)) : cont;
        }
    }
    return v[0];
}

}  // namespace ope
