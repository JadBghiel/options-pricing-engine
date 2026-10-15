// ope._core: the pybind11 bindings, thin on purpose, src/ope/cpp.py does the broadcasting and
// gives these the same signatures as the python functions
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "ope/black_scholes.hpp"
#include "ope/monte_carlo.hpp"
#include "ope/parallel.hpp"
#include "ope/philox.hpp"
#include "ope/tree.hpp"

namespace py = pybind11;

namespace {

// c contiguous float64, anything else (a list, an int array) is converted on the way in
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::span<const double> view(const Array& a) {
    if (a.ndim() != 1) {
        throw std::invalid_argument("expected 1d arrays, ope.cpp flattens them before the call");
    }
    return {a.data(), static_cast<std::size_t>(a.size())};
}

std::span<double> view(Array& a) {
    return {a.mutable_data(), static_cast<std::size_t>(a.size())};
}

ope::Chain chain(const Array& S, const Array& K, const Array& T, const Array& r,
                 const Array& sigma, const Array& q) {
    return {view(S), view(K), view(T), view(r), view(sigma), view(q)};
}

using MCTuple = std::tuple<double, double, std::int64_t>;

MCTuple as_tuple(const ope::MCResult& res) {
    return {res.estimate, res.std_error, res.n_paths};
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "c++ core of the options pricing engine, use it through ope.cpp";

    // so a benchmark can refuse a debug build
    m.attr("build_type") = OPE_BUILD_TYPE;
    m.attr("compiler") = OPE_COMPILER;
    m.attr("mc_chunk") = ope::mc_chunk;
    m.def("hardware_threads", [] { return ope::resolve_threads(0); });

    m.def("bs_price",
          [](const std::string& option, const Array& S, const Array& K, const Array& T,
             const Array& r, const Array& sigma, const Array& q, int threads) {
              const ope::Option opt = ope::parse_option(option);
              const ope::Chain c = chain(S, K, T, r, sigma, q);
              Array out(static_cast<py::ssize_t>(c.size()));
              std::span<double> o = view(out);
              // LEARNING NOTE: the GIL (python's global lock) is released for the loop, the
              // spans point into numpy buffers we hold a reference to -> no python object is
              // touched until the lock comes back, and other python threads run meanwhile,
              // an exception thrown in here reacquires it on the way out
              {
                  py::gil_scoped_release release;
                  ope::bs_price_chain(opt, c, o, threads);
              }
              return out;
          });

    m.def("bs_greeks", [](const std::string& option, const Array& S, const Array& K,
                          const Array& T, const Array& r, const Array& sigma, const Array& q,
                          int threads) {
        const ope::Option opt = ope::parse_option(option);
        const ope::Chain c = chain(S, K, T, r, sigma, q);
        const auto n = static_cast<py::ssize_t>(c.size());
        Array delta(n), gamma(n), vega(n), theta(n), rho(n);
        const ope::GreeksOut out{view(delta), view(gamma), view(vega), view(theta), view(rho)};
        {
            py::gil_scoped_release release;
            ope::bs_greeks_chain(opt, c, out, threads);
        }
        return py::make_tuple(delta, gamma, vega, theta, rho);
    });

    m.def("mc_price",
          [](const std::string& option, double S, double K, double T, double r, double sigma,
             double q, std::int64_t n_paths, bool antithetic, bool control_variate,
             std::uint64_t seed, int threads) {
              const ope::Option opt = ope::parse_option(option);
              py::gil_scoped_release release;
              return as_tuple(ope::mc_price(opt, S, K, T, r, sigma, q, n_paths, antithetic,
                                            control_variate, seed, threads));
          });

    m.def("mc_price_from_normals",
          [](const std::string& option, double S, double K, double T, double r, double sigma,
             double q, const Array& z, bool antithetic, bool control_variate) {
              const ope::Option opt = ope::parse_option(option);
              const std::span<const double> draws = view(z);
              py::gil_scoped_release release;
              return as_tuple(ope::mc_price_from_normals(opt, S, K, T, r, sigma, q, draws,
                                                         antithetic, control_variate));
          });

    m.def("normals", [](std::int64_t n, std::uint64_t seed, std::uint64_t first) {
        if (n < 0) {
            throw std::invalid_argument("n must be >= 0");
        }
        Array out(static_cast<py::ssize_t>(n));
        std::span<double> o = view(out);
        {
            py::gil_scoped_release release;
            ope::fill_normals(o, seed, first);
        }
        return out;
    });

    // raw generator, for the test that compares it w/ numpy's Philox bit for bit
    m.def("philox4x64", [](const ope::PhiloxBlock& counter, const ope::PhiloxKey& key) {
        return ope::philox4x64(counter, key);
    });

    m.def("crr_price", [](const std::string& option, double S, double K, double T, double r,
                          double sigma, double q, int steps, bool american) {
        const ope::Option opt = ope::parse_option(option);
        py::gil_scoped_release release;
        return ope::crr_price(opt, S, K, T, r, sigma, q, steps, american);
    });
}
