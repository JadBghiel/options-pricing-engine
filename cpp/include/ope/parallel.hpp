// thread count and a tiny parallel for, std::thread only so no openmp dependency
#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ope {

// 0 -> every hardware thread, n -> n. never less than 1
inline int resolve_threads(int threads) {
    if (threads < 0) {
        throw std::invalid_argument("threads must be >= 0 (0 = all cores)");
    }
    if (threads == 0) {
        return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    }
    return threads;
}

// splits [0, n) into contiguous ranges and calls f(begin, end) on each, 1 range per thread.
// below min_per_thread items per thread it uses fewer threads, starting a thread costs ~10us
template <class F>
void parallel_for(std::size_t n, int threads, std::size_t min_per_thread, F&& f) {
    const std::size_t by_size = std::max<std::size_t>(1, n / std::max<std::size_t>(1, min_per_thread));
    const std::size_t workers = std::min(static_cast<std::size_t>(resolve_threads(threads)), by_size);
    if (workers <= 1) {
        f(std::size_t{0}, n);
        return;
    }
    std::vector<std::jthread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t begin = n * w / workers;
        const std::size_t end = n * (w + 1) / workers;
        pool.emplace_back([&f, begin, end] { f(begin, end); });
    }
}  // jthread joins in its destructor

}  // namespace ope
