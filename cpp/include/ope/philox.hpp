// philox 4x64-10, the counter based generator of Salmon et al. 2011 (Random123), also in numpy
//
// a counter based generator is a keyed hash: block = philox(counter, key), no state to carry
// from 1 draw to the next -> any thread can jump to any position in the stream for free
#pragma once

#include <array>
#include <cstdint>

namespace ope {

using PhiloxBlock = std::array<std::uint64_t, 4>;
using PhiloxKey = std::array<std::uint64_t, 2>;

namespace detail {

__extension__ using u128 = unsigned __int128;  // gcc and clang, __extension__ keeps -Wpedantic quiet

constexpr std::uint64_t philox_m0 = 0xD2E7470EE14C6C93ULL;
constexpr std::uint64_t philox_m1 = 0xCA5A826395121157ULL;
constexpr std::uint64_t philox_w0 = 0x9E3779B97F4A7C15ULL;  // golden ratio, the key schedule
constexpr std::uint64_t philox_w1 = 0xBB67AE8584CAA73BULL;  // sqrt(3) - 1

}  // namespace detail

// 10 rounds, each 2 64x64 -> 128 bit multiplies whose high and low halves get mixed
constexpr PhiloxBlock philox4x64(PhiloxBlock ctr, PhiloxKey key) noexcept {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key[0] += detail::philox_w0;
            key[1] += detail::philox_w1;
        }
        const detail::u128 p0 = static_cast<detail::u128>(detail::philox_m0) * ctr[0];
        const detail::u128 p1 = static_cast<detail::u128>(detail::philox_m1) * ctr[2];
        const auto hi0 = static_cast<std::uint64_t>(p0 >> 64);
        const auto lo0 = static_cast<std::uint64_t>(p0);
        const auto hi1 = static_cast<std::uint64_t>(p1 >> 64);
        const auto lo1 = static_cast<std::uint64_t>(p1);
        ctr = {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
    }
    return ctr;
}

}  // namespace ope
