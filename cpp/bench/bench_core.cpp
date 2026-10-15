// c++ side of the benchmarks (google benchmark), run by benchmarks/run.sh
//
// same 3 workloads as benchmarks/test_bench.py, but w/o python in the way, and the monte carlo
// and the chain at 1, 2, 4 and 8 threads to measure the scaling
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "ope/black_scholes.hpp"
#include "ope/monte_carlo.hpp"
#include "ope/tree.hpp"

namespace {

constexpr std::size_t n_options = 100'000;
constexpr std::int64_t n_paths = 1'000'000;
constexpr int steps = 2'000;

struct ChainData {
    std::vector<double> S, K, T, r, sigma, q, out;
    ChainData()
        : S(n_options, 100.0), K(n_options), T(n_options), r(n_options, 0.03), sigma(n_options),
          q(n_options, 0.0), out(n_options) {
        std::mt19937_64 gen(0);
        std::uniform_real_distribution<double> strike(50, 150), mat(0.1, 2.0), vol(0.1, 0.6);
        for (std::size_t i = 0; i < n_options; ++i) {
            K[i] = strike(gen);
            T[i] = mat(gen);
            sigma[i] = vol(gen);
        }
    }
};

void BM_BlackScholesChain(benchmark::State& state) {
    static ChainData d;
    const ope::Chain chain{d.S, d.K, d.T, d.r, d.sigma, d.q};
    const auto threads = static_cast<int>(state.range(0));
    for (auto _ : state) {
        ope::bs_price_chain(ope::Option::call, chain, d.out, threads);
        benchmark::DoNotOptimize(d.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_options));
}

void BM_MonteCarlo(benchmark::State& state) {
    const auto threads = static_cast<int>(state.range(0));
    for (auto _ : state) {
        auto res = ope::mc_price(ope::Option::call, 42, 40, 0.5, 0.1, 0.2, 0, n_paths, false,
                                 false, 0, threads);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations() * n_paths);
}

void BM_AmericanTree(benchmark::State& state) {
    for (auto _ : state) {
        double v = ope::crr_price(ope::Option::put, 50, 50, 5.0 / 12, 0.1, 0.4, 0, steps, true);
        benchmark::DoNotOptimize(v);
    }
}

}  // namespace

// UseRealTime: the default clock is the cpu time of the main thread, which hardly moves while
// worker threads do the job -> a threaded run would look free
BENCHMARK(BM_BlackScholesChain)->ArgName("threads")->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MonteCarlo)->ArgName("threads")->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AmericanTree)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
