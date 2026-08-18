// black scholes merton closed form, same formulas and conventions as src/ope/black_scholes.py
//
// units: r, q and sigma annualized and continuously compounded, T in years. greeks in Hull's
// units: vega and rho per 1.00, theta per year
#pragma once

#include <cstddef>
#include <span>

#include "ope/option.hpp"

namespace ope {

struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// 1 option. throws std::invalid_argument on bad inputs, same msgs as python
double bs_price(Option option, double S, double K, double T, double r, double sigma, double q = 0.0);
Greeks bs_greeks(Option option, double S, double K, double T, double r, double sigma, double q = 0.0);

// a whole chain, every span the same length. inputs are all checked before any thread starts
struct Chain {
    std::span<const double> S, K, T, r, sigma, q;
    std::size_t size() const;
};

void bs_price_chain(Option option, const Chain& chain, std::span<double> out, int threads = 1);

struct GreeksOut {
    std::span<double> delta, gamma, vega, theta, rho;
};

void bs_greeks_chain(Option option, const Chain& chain, const GreeksOut& out, int threads = 1);

double norm_cdf(double x);
double norm_pdf(double x);

}  // namespace ope
