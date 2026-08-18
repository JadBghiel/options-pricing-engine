// cox ross rubinstein binomial tree, european and american, same as src/ope/trees.py
#pragma once

#include "ope/option.hpp"

namespace ope {

// throws std::invalid_argument on bad inputs or when p leaves (0, 1), same msgs as python
double crr_price(Option option, double S, double K, double T, double r, double sigma, double q,
                 int steps, bool american);

}  // namespace ope
