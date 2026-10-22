// intx: extended precision integer library.
// Copyright 2019 Pawel Bylica.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <climits>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>   // fputs
#include <cstdlib>  // abort
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 5030)  // Allow unknown attributes.
    #pragma warning(disable : 4127)  // Allow constant conditions (warning removed in VS 2026).
#endif

#ifndef __has_builtin
    #define __has_builtin(NAME) 0
#endif

#ifndef __has_feature
    #define __has_feature(NAME) 0
#endif

#ifdef _MSC_VER
    #include <intrin.h>
#endif

#if __has_builtin(__builtin_expect)
    #define INTX_UNLIKELY(EXPR) __builtin_expect(bool{EXPR}, false)
#else
    #define INTX_UNLIKELY(EXPR) (bool{EXPR})
#endif

#ifndef NDEBUG
    #define INTX_REQUIRE assert
#else
    #define INTX_REQUIRE(X) (X) ? (void)0 : intx::unreachable()
#endif


// Detect compiler support for 128-bit integer __int128
#ifdef __SIZEOF_INT128__
    #define INTX_HAS_BUILTIN_INT128 1
#else
    #define INTX_HAS_BUILTIN_INT128 0
#endif

namespace intx
{
/// Mark a possible code path as unreachable (invokes undefined behavior).
/// TODO(C++23): Use std::unreachable().
[[noreturn]] inline void unreachable() noexcept
{
#if __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(false);
#endif
}

#if INTX_HAS_BUILTIN_INT128
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"  // Usage of __int128 triggers a pedantic warning.

/// Alias for the compiler supported unsigned __int128 type.
using builtin_uint128 = unsigned __int128;

    #pragma GCC diagnostic pop
#endif


template <unsigned N>
struct uint;

/// Contains result of add/sub/etc with a carry flag.
template <typename T>
struct result_with_carry
{
    T value;
    bool carry;

    /// Initializes the value and the carry.
    ///
    /// TODO(gcc-17): This makes the type a non-aggregate, which helps GCC codegen.
    /// https://gcc.gnu.org/PR110459
    constexpr result_with_carry(const T& v, bool c) noexcept : value{v}, carry{c} {}

    /// Conversion to tuple of references, to allow usage with std::tie().
    constexpr explicit(false) operator std::tuple<T&, bool&>() noexcept { return {value, carry}; }
};

template <typename QuotT, typename RemT = QuotT>
struct div_result
{
    QuotT quot;
    RemT rem;

    bool operator==(const div_result&) const = default;

    /// Conversion to tuple of references, to allow usage with std::tie().
    constexpr explicit(false) operator std::tuple<QuotT&, RemT&>() noexcept { return {quot, rem}; }
};

/// Addition with carry.
constexpr result_with_carry<uint64_t> addc(uint64_t x, uint64_t y, bool carry = false) noexcept
{
    // On rv32im, __builtin_addcll generates ~30 instructions with branches because
    // GCC's 64-bit carry detection on 32-bit targets uses conditional branches.
    // Manual 32-bit decomposition compiles to ~14 branchless add/sltu instructions.
#if defined(__riscv) && __riscv_xlen == 32
    const auto x_lo = static_cast<uint32_t>(x);
    const auto x_hi = static_cast<uint32_t>(x >> 32);
    const auto y_lo = static_cast<uint32_t>(y);
    const auto y_hi = static_cast<uint32_t>(y >> 32);

    auto lo = x_lo + y_lo;
    uint32_t c = (lo < x_lo);
    lo += carry;
    c += (lo < static_cast<uint32_t>(carry));
    auto hi = x_hi + y_hi;
    uint32_t cout = (hi < x_hi);
    hi += c;
    cout += (hi < c);

    return {(static_cast<uint64_t>(hi) << 32) | lo, cout != 0};
#elif __has_builtin(__builtin_addcll)
    if (!std::is_constant_evaluated())
    {
        unsigned long long carryout = 0;  // NOLINT(google-runtime-int)
        const auto s = __builtin_addcll(x, y, carry, &carryout);
        static_assert(sizeof(s) == sizeof(uint64_t));
        return {s, static_cast<bool>(carryout)};
    }
#elif __has_builtin(__builtin_ia32_addcarryx_u64)
    if (!std::is_constant_evaluated())
    {
        unsigned long long s = 0;  // NOLINT(google-runtime-int)
        static_assert(sizeof(s) == sizeof(uint64_t));
        const auto carryout = __builtin_ia32_addcarryx_u64(carry, x, y, &s);
        return {s, static_cast<bool>(carryout)};
    }
#endif

#if !defined(__riscv) || __riscv_xlen != 32
    const auto s = x + y;
    const auto carry1 = s < x;
    const auto t = s + carry;
    const auto carry2 = t < s;
    return {t, carry1 || carry2};
#endif
}

/// Subtraction with carry (borrow).
constexpr result_with_carry<uint64_t> subc(uint64_t x, uint64_t y, bool carry = false) noexcept
{
    // On rv32im, __builtin_subcll generates ~30 instructions with branches.
    // Manual 32-bit decomposition compiles to ~14 branchless sub/sltu instructions.
#if defined(__riscv) && __riscv_xlen == 32
    const auto x_lo = static_cast<uint32_t>(x);
    const auto x_hi = static_cast<uint32_t>(x >> 32);
    const auto y_lo = static_cast<uint32_t>(y);
    const auto y_hi = static_cast<uint32_t>(y >> 32);

    auto lo = x_lo - y_lo;
    uint32_t b = (x_lo < y_lo);
    const auto lo2 = lo - carry;
    b += (lo < static_cast<uint32_t>(carry));
    auto hi = x_hi - y_hi;
    uint32_t bout = (x_hi < y_hi);
    const auto hi2 = hi - b;
    bout += (hi < b);

    return {(static_cast<uint64_t>(hi2) << 32) | lo2, bout != 0};
// Use __builtin_subcll if available (except buggy Xcode 14.3.1 on arm64).
#elif __has_builtin(__builtin_subcll) && __apple_build_version__ != 14030022
    if (!std::is_constant_evaluated())
    {
        unsigned long long carryout = 0;  // NOLINT(google-runtime-int)
        const auto d = __builtin_subcll(x, y, carry, &carryout);
        static_assert(sizeof(d) == sizeof(uint64_t));
        return {d, static_cast<bool>(carryout)};
    }
#elif __has_builtin(__builtin_ia32_sbb_u64)
    if (!std::is_constant_evaluated())
    {
        unsigned long long d = 0;  // NOLINT(google-runtime-int)
        static_assert(sizeof(d) == sizeof(uint64_t));
        const auto carryout = __builtin_ia32_sbb_u64(carry, x, y, &d);
        return {d, static_cast<bool>(carryout)};
    }
#endif

#if !defined(__riscv) || __riscv_xlen != 32
    const auto d = x - y;
    const auto carry1 = x < y;
    const auto e = d - carry;
    const auto carry2 = d < uint64_t{carry};
    return {e, carry1 || carry2};
#endif
}

/// Addition with carry.
template <unsigned N>
constexpr result_with_carry<uint<N>> addc(
    const uint<N>& x, const uint<N>& y, bool carry = false) noexcept
{
    uint<N> s;
    bool k = carry;
    for (size_t i = 0; i < uint<N>::num_words; ++i)
    {
        auto t = addc(x[i], y[i], k);
        s[i] = t.value;
        k = t.carry;
    }
    return {s, k};
}

/// Performs subtraction of two unsigned numbers and returns the difference
/// and the carry bit (aka borrow, overflow).
template <unsigned N>
constexpr result_with_carry<uint<N>> subc(
    const uint<N>& x, const uint<N>& y, bool carry = false) noexcept
{
    uint<N> z;
    bool k = carry;
    for (size_t i = 0; i < uint<N>::num_words; ++i)
    {
        auto t = subc(x[i], y[i], k);
        z[i] = t.value;
        k = t.carry;
    }
    return {z, k};
}

constexpr uint<128> umul(uint64_t x, uint64_t y) noexcept;


/// The 128-bit unsigned integer.
///
/// This type is defined as a specialization of uint<> to easier integration with full intx package,
/// however, uint128 may be used independently.
template <>
struct uint<128>
{
    using word_type = uint64_t;
    static constexpr auto word_num_bits = sizeof(word_type) * 8;
    static constexpr unsigned num_bits = 128;
    static constexpr auto num_words = num_bits / word_num_bits;

private:
    uint64_t words_[2]{};

public:
    constexpr uint() noexcept = default;

    constexpr uint(uint64_t low, uint64_t high) noexcept : words_{low, high} {}

    template <typename T>
    constexpr explicit(false) uint(T x) noexcept
        requires std::is_convertible_v<T, uint64_t>
      : words_{static_cast<uint64_t>(x), 0}
    {}

    /// Constructs from words with words[0] being the least significant word.
    /// The size of the span must be less than or equal to num_words.
    constexpr explicit uint(std::span<const uint64_t> words) noexcept
    {
        INTX_REQUIRE(words.size() <= num_words);
        std::ranges::copy(words, words_);
    }

#if INTX_HAS_BUILTIN_INT128
    constexpr explicit(false) uint(builtin_uint128 x) noexcept
      : words_{uint64_t(x), uint64_t(x >> 64)}
    {}

    constexpr explicit operator builtin_uint128() const noexcept
    {
        return (builtin_uint128{words_[1]} << 64) | words_[0];
    }
#endif

    constexpr uint64_t& operator[](size_t i) noexcept { return words_[i]; }
    constexpr const uint64_t& operator[](size_t i) const noexcept { return words_[i]; }

    constexpr explicit operator bool() const noexcept { return (words_[0] | words_[1]) != 0; }

    /// Explicit converting operator for all builtin integral types.
    template <typename Int>
    constexpr explicit operator Int() const noexcept
        requires std::is_integral_v<Int>
    {
        return static_cast<Int>(words_[0]);
    }

    friend constexpr uint operator+(uint x, uint y) noexcept
    {
#if INTX_HAS_BUILTIN_INT128 && defined(__GNUC__) && !defined(__clang__)
        // TODO(gcc): GCC splits chains of addc() into setc and extra adds. The builtin type
        // keeps add/adc together. https://gcc.gnu.org/PR79173
        return builtin_uint128{x} + builtin_uint128{y};
#else
        return addc(x, y).value;
#endif
    }

    constexpr uint operator+() const noexcept { return *this; }

    friend constexpr uint operator-(uint x, uint y) noexcept
    {
#if INTX_HAS_BUILTIN_INT128 && defined(__GNUC__) && !defined(__clang__)
        // TODO(gcc): See operator+.
        return builtin_uint128{x} - builtin_uint128{y};
#else
        return subc(x, y).value;
#endif
    }

    constexpr uint operator-() const noexcept
    {
        // Implementing as subtraction is better than ~x + 1.
        // Clang9: Perfect.
        // GCC8: Does something weird.
        return 0 - *this;
    }

    constexpr uint& operator+=(uint y) noexcept { return *this = *this + y; }

    constexpr uint& operator-=(uint y) noexcept { return *this = *this - y; }

    constexpr uint& operator++() noexcept { return *this += 1; }

    constexpr uint& operator--() noexcept { return *this -= 1; }

    constexpr const uint operator++(int) noexcept  // NOLINT(*-const-return-type)
    {
        const auto ret = *this;
        *this += 1;
        return ret;
    }

    constexpr const uint operator--(int) noexcept  // NOLINT(*-const-return-type)
    {
        const auto ret = *this;
        *this -= 1;
        return ret;
    }

    friend constexpr bool operator==(uint x, uint y) noexcept
    {
        return ((x[0] ^ y[0]) | (x[1] ^ y[1])) == 0;
    }

    friend constexpr bool operator<(uint x, uint y) noexcept
    {
        // OPT: This should be implemented by checking the borrow of x - y,
        //      but compilers (GCC8, Clang7)
        //      have problem with properly optimizing subtraction.

#if INTX_HAS_BUILTIN_INT128
        return builtin_uint128{x} < builtin_uint128{y};
#else
        return (unsigned{x[1] < y[1]} | (unsigned{x[1] == y[1]} & unsigned{x[0] < y[0]})) != 0;
#endif
    }
    friend constexpr bool operator<=(uint x, uint y) noexcept { return !(y < x); }
    friend constexpr bool operator>(uint x, uint y) noexcept { return y < x; }
    friend constexpr bool operator>=(uint x, uint y) noexcept { return !(x < y); }

    friend constexpr std::strong_ordering operator<=>(uint x, uint y) noexcept
    {
        if (x == y)
            return std::strong_ordering::equal;

        return (x < y) ? std::strong_ordering::less : std::strong_ordering::greater;
    }

    friend constexpr uint operator~(uint x) noexcept { return {~x[0], ~x[1]}; }
    friend constexpr uint operator|(uint x, uint y) noexcept { return {x[0] | y[0], x[1] | y[1]}; }
    friend constexpr uint operator&(uint x, uint y) noexcept { return {x[0] & y[0], x[1] & y[1]}; }
    friend constexpr uint operator^(uint x, uint y) noexcept { return {x[0] ^ y[0], x[1] ^ y[1]}; }

    friend constexpr uint operator<<(uint x, uint64_t shift) noexcept
    {
        if (shift < 64)
        {
            // Find the part moved from lo to hi.
            // For shift == 0 right shift by (64 - shift) is invalid so
            // split it into 2 shifts by 1 and (63 - shift).
            return {x[0] << shift, (x[1] << shift) | ((x[0] >> 1) >> (63 - shift))};
        }
        if (shift < 128)
        {
            // The lo part becomes the shifted hi part.
            return {0, x[0] << (shift - 64)};
        }

        // Guarantee "defined" behavior for shifts larger than 128.
        return 0;
    }

    friend constexpr uint operator<<(uint x, std::integral auto shift) noexcept
    {
        static_assert(sizeof(shift) <= sizeof(uint64_t));
        return x << static_cast<uint64_t>(shift);
    }

    friend constexpr uint operator<<(uint x, uint shift) noexcept
    {
        if (shift[1] != 0) [[unlikely]]
            return 0;

        return x << shift[0];
    }

    friend constexpr uint operator>>(uint x, uint64_t shift) noexcept
    {
        if (shift < 64)
        {
            // Find the part moved from lo to hi.
            // For shift == 0 left shift by (64 - shift) is invalid so
            // split it into 2 shifts by 1 and (63 - shift).
            return {(x[0] >> shift) | ((x[1] << 1) << (63 - shift)), x[1] >> shift};
        }
        if (shift < 128)
        {
            // The lo part becomes the shifted hi part.
            return {x[1] >> (shift - 64), 0};
        }

        // Guarantee "defined" behavior for shifts larger than 128.
        return 0;
    }

    friend constexpr uint operator>>(uint x, std::integral auto shift) noexcept
    {
        static_assert(sizeof(shift) <= sizeof(uint64_t));
        return x >> static_cast<uint64_t>(shift);
    }

    friend constexpr uint operator>>(uint x, uint shift) noexcept
    {
        if (shift[1] != 0) [[unlikely]]
            return 0;

        return x >> shift[0];
    }

    friend constexpr uint operator*(uint x, uint y) noexcept
    {
        auto p = umul(x[0], y[0]);
        p[1] += (x[0] * y[1]) + (x[1] * y[0]);
        return {p[0], p[1]};
    }

    friend constexpr div_result<uint> udivrem(uint x, uint y) noexcept;
    friend constexpr uint operator/(uint x, uint y) noexcept { return udivrem(x, y).quot; }
    friend constexpr uint operator%(uint x, uint y) noexcept { return udivrem(x, y).rem; }

    constexpr uint& operator*=(uint y) noexcept { return *this = *this * y; }
    constexpr uint& operator|=(uint y) noexcept { return *this = *this | y; }
    constexpr uint& operator&=(uint y) noexcept { return *this = *this & y; }
    constexpr uint& operator^=(uint y) noexcept { return *this = *this ^ y; }
    constexpr uint& operator<<=(uint shift) noexcept { return *this = *this << shift; }
    constexpr uint& operator>>=(uint shift) noexcept { return *this = *this >> shift; }
    constexpr uint& operator/=(uint y) noexcept { return *this = *this / y; }
    constexpr uint& operator%=(uint y) noexcept { return *this = *this % y; }
};

using uint128 = uint<128>;


/// Full unsigned multiplication 64 x 64 -> 128.
constexpr uint128 umul(uint64_t x, uint64_t y) noexcept
{
#if defined(__riscv) && __riscv_xlen == 32
    // rv32im: use uint32_t operands so GCC emits native mul+mulhu pairs
    // (4 pairs for the cross-products) instead of __muldi3/__multi3 libcalls.
    if (!std::is_constant_evaluated())
    {
        const auto xl = static_cast<uint32_t>(x);
        const auto xh = static_cast<uint32_t>(x >> 32);
        const auto yl = static_cast<uint32_t>(y);
        const auto yh = static_cast<uint32_t>(y >> 32);

        // Each uint64_t(u32) * uint64_t(u32) compiles to mul+mulhu on rv32im.
        const uint64_t t0 = static_cast<uint64_t>(xl) * yl;
        const uint64_t t1 = static_cast<uint64_t>(xh) * yl;
        const uint64_t t2 = static_cast<uint64_t>(xl) * yh;
        const uint64_t t3 = static_cast<uint64_t>(xh) * yh;

        const uint64_t u1 = t1 + static_cast<uint32_t>(t0 >> 32);
        const uint64_t u2 = t2 + static_cast<uint32_t>(u1);

        const uint64_t lo =
            (static_cast<uint64_t>(static_cast<uint32_t>(u2)) << 32) |
            static_cast<uint32_t>(t0);
        const uint64_t hi = t3 + (u2 >> 32) + (u1 >> 32);
        return {lo, hi};
    }
    // constexpr: fall through to portable path below.
#elif INTX_HAS_BUILTIN_INT128
    return builtin_uint128{x} * builtin_uint128{y};
#elif defined(_MSC_VER) && _MSC_VER >= 1925 && defined(_M_X64)
    if (!std::is_constant_evaluated())
    {
        unsigned __int64 hi = 0;
        const auto lo = _umul128(x, y, &hi);
        return {lo, hi};
    }
    // For constexpr fallback to the portable variant.
#endif

    // Portable full unsigned multiplication 64 x 64 -> 128.
    uint64_t xlo = x & 0xffffffff;
    uint64_t xhi = x >> 32;
    uint64_t ylo = y & 0xffffffff;
    uint64_t yhi = y >> 32;

    uint64_t t0 = xlo * ylo;
    uint64_t t1 = xhi * ylo;
    uint64_t t2 = xlo * yhi;
    uint64_t t3 = xhi * yhi;

    uint64_t u1 = t1 + (t0 >> 32);
    uint64_t u2 = t2 + (u1 & 0xffffffff);

    uint64_t lo = (u2 << 32) | (t0 & 0xffffffff);
    uint64_t hi = t3 + (u2 >> 32) + (u1 >> 32);
    return {lo, hi};
}

constexpr uint64_t bit_test(uint64_t x, size_t bit_index) noexcept
{
    // This pattern matches BT instruction on x86.
    // On architectures without dedicated instruction,
    // this is likely converted to (x >> bit_index) & 1.
    return (x & (uint64_t{1} << bit_index)) != 0;
}

constexpr unsigned clz(std::unsigned_integral auto x) noexcept
{
    return static_cast<unsigned>(std::countl_zero(x));
}

constexpr unsigned ctz(std::unsigned_integral auto x) noexcept
{
    return static_cast<unsigned>(std::countr_zero(x));
}

/// Counts the number of bits needed to represent the value. For 0 returns 0.
///
/// This works like std::bit_width, but returns unsigned.
constexpr unsigned bit_width(std::unsigned_integral auto x) noexcept
{
    return static_cast<unsigned>(std::numeric_limits<decltype(x)>::digits) - clz(x);
}

/// Funnel shift left: returns the high word of the 128-bit value hi:lo shifted left by shift.
/// The shift must be less than 64.
constexpr uint64_t fshl(uint64_t hi, uint64_t lo, unsigned shift) noexcept
{
    INTX_REQUIRE(shift < 64);
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
    // TODO(gcc): GCC does not emit SHLD for funnel shifts. https://gcc.gnu.org/PR82261
    if (!std::is_constant_evaluated())
    {
        asm("shld{q %b[s], %[lo], %[hi]| %[hi], %[lo], %b[s]}"
            : [hi] "+rm"(hi)
            : [lo] "r"(lo), [s] "cJ"(shift)
            : "cc");
        return hi;
    }
#endif
    return shift == 0 ? hi : (hi << shift) | (lo >> (64 - shift));
}

/// Funnel shift right: returns the low word of the 128-bit value hi:lo shifted right by shift.
/// The shift must be less than 64.
constexpr uint64_t fshr(uint64_t hi, uint64_t lo, unsigned shift) noexcept
{
    INTX_REQUIRE(shift < 64);
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
    // TODO(gcc): GCC does not emit SHRD for funnel shifts. https://gcc.gnu.org/PR82261
    if (!std::is_constant_evaluated())
    {
        asm("shrd{q %b[s], %[hi], %[lo]| %[lo], %[hi], %b[s]}"
            : [lo] "+rm"(lo)
            : [hi] "r"(hi), [s] "cJ"(shift)
            : "cc");
        return lo;
    }
#endif
    return shift == 0 ? lo : (lo >> shift) | (hi << (64 - shift));
}

/// Left-shifts a little-endian word array x by shift bits and stores the result in r.
/// The r must have the same size as x, or one word more to also receive the bits shifted out.
/// The r may alias x. The shift must be less than 64.
template <size_t RN, size_t XN>
constexpr void shl(
    std::span<uint64_t, RN> r, std::span<const uint64_t, XN> x, unsigned shift) noexcept
{
    if constexpr (RN != std::dynamic_extent && XN != std::dynamic_extent)
        static_assert(XN != 0 && (RN == XN || RN == XN + 1));
    INTX_REQUIRE(!x.empty() && (r.size() == x.size() || r.size() == x.size() + 1));
    INTX_REQUIRE(shift < 64);
    const auto n = x.size();
    if (r.size() > n)
        r[n] = fshl(0, x[n - 1], shift);
    for (size_t i = n - 1; i != 0; --i)
        r[i] = fshl(x[i], x[i - 1], shift);
    r[0] = x[0] << shift;
}

/// Right-shifts a little-endian word array x by shift bits and stores the result in r
/// of the same size. The bits shifted out of x are lost. The r may alias x.
/// The shift must be less than 64.
constexpr void shr(std::span<uint64_t> r, std::span<const uint64_t> x, unsigned shift) noexcept
{
    INTX_REQUIRE(!x.empty() && r.size() == x.size());
    INTX_REQUIRE(shift < 64);
    for (size_t i = 1; i < x.size(); ++i)
        r[i - 1] = fshr(x[i], x[i - 1], shift);
    r[x.size() - 1] = x[x.size() - 1] >> shift;
}

constexpr unsigned clz(uint128 x) noexcept
{
    // In this order `h == 0` we get fewer instructions than in the case of `h != 0`.
    return x[1] == 0 ? clz(x[0]) + 64 : clz(x[1]);
}

template <typename T>
T bswap(T x) noexcept = delete;  // Disable type auto promotion

constexpr uint8_t bswap(uint8_t x) noexcept
{
    return x;
}

constexpr uint16_t bswap(uint16_t x) noexcept
{
#if __has_builtin(__builtin_bswap16)
    return __builtin_bswap16(x);
#else
    #ifdef _MSC_VER
    if (!std::is_constant_evaluated())
        return _byteswap_ushort(x);
    #endif
    return static_cast<uint16_t>((x << 8) | (x >> 8));
#endif
}

constexpr uint32_t bswap(uint32_t x) noexcept
{
    // On rv32im without Zbb, __builtin_bswap32 generates a __bswapsi2 libcall.
    // Use inline shift-and-mask that compiles to ~8 instructions.
#if defined(__riscv) && __riscv_xlen == 32 && !defined(__riscv_zbb)
    if (!std::is_constant_evaluated())
    {
        const auto a = ((x << 8) & 0xFF00FF00u) | ((x >> 8) & 0x00FF00FFu);
        return (a << 16) | (a >> 16);
    }
#endif
#if __has_builtin(__builtin_bswap32)
    return __builtin_bswap32(x);
#else
    #ifdef _MSC_VER
    if (!std::is_constant_evaluated())
        return _byteswap_ulong(x);
    #endif
    const auto a = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0x00FF00FF);
    return (a << 16) | (a >> 16);
#endif
}

constexpr uint64_t bswap(uint64_t x) noexcept
{
    // On rv32im without Zbb, __builtin_bswap64 generates a __bswapdi2 libcall.
    // Decompose into two inline bswap32 + half-swap: ~18 instructions vs libcall.
#if defined(__riscv) && __riscv_xlen == 32 && !defined(__riscv_zbb)
    if (!std::is_constant_evaluated())
    {
        const auto lo = static_cast<uint32_t>(x);
        const auto hi = static_cast<uint32_t>(x >> 32);
        return (static_cast<uint64_t>(bswap(lo)) << 32) | bswap(hi);
    }
#endif
#if __has_builtin(__builtin_bswap64)
    return __builtin_bswap64(x);
#else
    #ifdef _MSC_VER
    if (!std::is_constant_evaluated())
        return _byteswap_uint64(x);
    #endif
    const auto a = ((x << 8) & 0xFF00FF00FF00FF00) | ((x >> 8) & 0x00FF00FF00FF00FF);
    const auto b = ((a << 16) & 0xFFFF0000FFFF0000) | ((a >> 16) & 0x0000FFFF0000FFFF);
    return (b << 32) | (b >> 32);
#endif
}

constexpr uint128 bswap(uint128 x) noexcept
{
    return {bswap(x[1]), bswap(x[0])};
}


/// Division.
/// @{

namespace internal
{
/// Reciprocal lookup table.
constexpr auto reciprocal_table = []() noexcept {
    std::array<uint16_t, 256> table{};
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = static_cast<uint16_t>(0x7fd00 / (i + 256));
    return table;
}();
}  // namespace internal

/// Computes the reciprocal (2^128 - 1) / d - 2^64 for normalized d.
///
/// Based on Algorithm 2 from "Improved division by invariant integers".
constexpr uint64_t reciprocal_2by1(uint64_t d) noexcept
{
    INTX_REQUIRE(d & 0x8000000000000000);  // Must be normalized.

    const uint64_t d9 = d >> 55;
    const uint32_t v0 = internal::reciprocal_table[static_cast<size_t>(d9 - 256)];

    const uint64_t d40 = (d >> 24) + 1;
    const uint64_t v1 = (v0 << 11) - uint32_t(uint32_t{v0 * v0} * d40 >> 40) - 1;

    const uint64_t v2 = (v1 << 13) + (v1 * (0x1000000000000000 - v1 * d40) >> 47);

    const uint64_t d0 = d & 1;
    const uint64_t d63 = (d >> 1) + d0;  // ceil(d/2)
    const uint64_t e = ((v2 >> 1) & (0 - d0)) - (v2 * d63);
    const uint64_t v3 = (umul(v2, e)[1] >> 1) + (v2 << 31);

    const uint64_t v4 = v3 - (umul(v3, d) + d)[1] - d;
    return v4;
}

constexpr uint64_t reciprocal_3by2(uint128 d) noexcept
{
    auto v = reciprocal_2by1(d[1]);
    auto p = d[1] * v;
    p += d[0];
    if (p < d[0])
    {
        --v;
        if (p >= d[1])
        {
            --v;
            p -= d[1];
        }
        p -= d[1];
    }

    const auto t = umul(v, d[0]);

    p += t[1];
    if (p < t[1])
    {
        --v;
        if (p >= d[1])
        {
            if (p > d[1] || t[0] >= d[0])
                --v;
        }
    }
    return v;
}

constexpr div_result<uint64_t> udivrem_2by1(uint128 u, uint64_t d, uint64_t v) noexcept
{
    auto q = umul(v, u[1]);
    q += u;

    ++q[1];

    auto r = u[0] - (q[1] * d);

    if (r > q[0])
    {
        --q[1];
        r += d;
    }

    if (r >= d)
    {
        ++q[1];
        r -= d;
    }

    return {q[1], r};
}

constexpr div_result<uint64_t, uint128> udivrem_3by2(
    uint64_t u2, uint64_t u1, uint64_t u0, uint128 d, uint64_t v) noexcept
{
    auto q = umul(v, u2);
    q += uint128{u1, u2};

    auto r1 = u1 - (q[1] * d[1]);

    auto t = umul(d[0], q[1]);

    auto r = uint128{u0, r1} - t - d;
    r1 = r[1];

    ++q[1];

    if (r1 >= q[0])
    {
        --q[1];
        r += d;
    }

    if (r >= d)
    {
        ++q[1];
        r -= d;
    }

    return {q[1], r};
}

constexpr div_result<uint128> udivrem(uint128 x, uint128 y) noexcept
{
    if (y[1] == 0)
    {
        INTX_REQUIRE(y[0] != 0);  // Division by 0.

        const auto lsh = clz(y[0]);
        const auto rsh = (64 - lsh) % 64;
        const auto rsh_mask = uint64_t{lsh == 0} - 1;

        const auto yn = y[0] << lsh;
        const auto xn_lo = x[0] << lsh;
        const auto xn_hi = (x[1] << lsh) | ((x[0] >> rsh) & rsh_mask);
        const auto xn_ex = (x[1] >> rsh) & rsh_mask;

        const auto v = reciprocal_2by1(yn);
        const auto res1 = udivrem_2by1({xn_hi, xn_ex}, yn, v);
        const auto res2 = udivrem_2by1({xn_lo, res1.rem}, yn, v);
        return {{res2.quot, res1.quot}, res2.rem >> lsh};
    }

    if (y[1] > x[1])
        return {0, x};

    const auto lsh = clz(y[1]);
    if (lsh == 0)
    {
        const auto q = unsigned{y[1] < x[1]} | unsigned{y[0] <= x[0]};
        return {q, x - (q ? y : 0)};
    }

    const auto rsh = 64 - lsh;

    const auto yn_lo = y[0] << lsh;
    const auto yn_hi = (y[1] << lsh) | (y[0] >> rsh);
    const auto xn_lo = x[0] << lsh;
    const auto xn_hi = (x[1] << lsh) | (x[0] >> rsh);
    const auto xn_ex = x[1] >> rsh;

    const auto v = reciprocal_3by2({yn_lo, yn_hi});
    const auto res = udivrem_3by2(xn_ex, xn_hi, xn_lo, {yn_lo, yn_hi}, v);

    return {res.quot, res.rem >> lsh};
}

constexpr div_result<uint128> sdivrem(uint128 x, uint128 y) noexcept
{
    constexpr auto sign_mask = uint128{1} << 127;
    const auto x_is_neg = (x & sign_mask) != 0;
    const auto y_is_neg = (y & sign_mask) != 0;

    const auto x_abs = x_is_neg ? -x : x;
    const auto y_abs = y_is_neg ? -y : y;

    const auto q_is_neg = x_is_neg ^ y_is_neg;

    const auto res = udivrem(x_abs, y_abs);

    return {q_is_neg ? -res.quot : res.quot, x_is_neg ? -res.rem : res.rem};
}

/// @}

}  // namespace intx


namespace std
{
template <unsigned N>
struct numeric_limits<intx::uint<N>>  // NOLINT(cert-dcl58-cpp)
{
    using type = intx::uint<N>;

    static constexpr bool is_specialized = true;
    static constexpr bool is_integer = true;
    static constexpr bool is_signed = false;
    static constexpr bool is_exact = true;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_round_style round_style = round_toward_zero;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = true;
    static constexpr int digits = CHAR_BIT * sizeof(type);
    static constexpr int digits10 = int(0.3010299956639812 * digits);
    static constexpr int max_digits10 = 0;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr bool traps = std::numeric_limits<unsigned>::traps;
    static constexpr bool tinyness_before = false;

    static constexpr type min() noexcept { return 0; }
    static constexpr type lowest() noexcept { return min(); }
    static constexpr type max() noexcept { return ~type{0}; }
    static constexpr type epsilon() noexcept { return 0; }
    static constexpr type round_error() noexcept { return 0; }
    static constexpr type infinity() noexcept { return 0; }
    static constexpr type quiet_NaN() noexcept { return 0; }
    static constexpr type signaling_NaN() noexcept { return 0; }
    static constexpr type denorm_min() noexcept { return 0; }
};
}  // namespace std

namespace intx
{
template <typename T>
[[noreturn]] inline void throw_(const char* what)
{
#if __cpp_exceptions
    throw T{what};
#else
    std::fputs(what, stderr);
    std::abort();
#endif
}

constexpr uint8_t from_dec_digit(char c)
{
    if (c < '0' || c > '9')
        throw_<std::invalid_argument>("invalid digit");
    return static_cast<uint8_t>(c - '0');
}

constexpr uint8_t from_hex_digit(char c)
{
    if (c >= 'a' && c <= 'f')
        return static_cast<uint8_t>(c - ('a' - 10));
    if (c >= 'A' && c <= 'F')
        return static_cast<uint8_t>(c - ('A' - 10));
    return from_dec_digit(c);
}

template <typename Int>
constexpr Int from_string(const char* str)
{
    auto s = str;
    auto x = Int{};
    size_t num_digits = 0;

    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    {
        s += 2;
        while (const auto c = *s++)
        {
            if (++num_digits > sizeof(x) * 2)
                throw_<std::out_of_range>(str);
            x = (x << uint64_t{4}) | from_hex_digit(c);
        }
        return x;
    }

    while (const auto c = *s++)
    {
        if (num_digits++ > std::numeric_limits<Int>::digits10)
            throw_<std::out_of_range>(str);

        const auto d = from_dec_digit(c);
        x = x * Int{10} + d;
        if (x < d)
            throw_<std::out_of_range>(str);
    }
    return x;
}

template <typename Int>
constexpr Int from_string(const std::string& s)
{
    return from_string<Int>(s.c_str());
}

template <unsigned N>
inline std::string to_string(uint<N> x, int base = 10)
{
    if (base < 2 || base > 36)
        throw_<std::invalid_argument>("invalid base");

    constexpr auto WORD_MAX = std::numeric_limits<uint64_t>::max();
    const auto b = static_cast<uint64_t>(base);
    // The digits, written from the end. The base 2 needs the most: N digits.
    std::array<char, N> s;  // NOLINT(cppcoreguidelines-pro-type-member-init)
    auto it = s.end();
    const auto put_digit = [&it, b](uint64_t& r) {
        *--it = "0123456789abcdefghijklmnopqrstuvwxyz"[r % b];
        r /= b;
    };

    if (x > WORD_MAX)
    {
        // Divide by the largest power of the base fitting in a word, base^digits: each division
        // produces a chunk of digits, until the rest fits in a word.
        auto power = b;
        size_t digits = 1;
        while (power <= WORD_MAX / b)
        {
            power *= b;
            ++digits;
        }
        const auto divisor = uint<N>{power};
        do
        {
            const auto res = udivrem(x, divisor);
            x = res.quot;
            auto r = static_cast<uint64_t>(res.rem);
            for (size_t i = 0; i < digits; ++i)
                put_digit(r);
        } while (x > WORD_MAX);
    }

    // The leading digits, at least one.
    auto r = static_cast<uint64_t>(x);
    do
        put_digit(r);
    while (r != 0);
    return {it, s.end()};
}

template <unsigned N>
inline std::string hex(uint<N> x)
{
    return to_string(x, 16);
}

template <unsigned N>
struct uint
{
    using word_type = uint64_t;
    static constexpr auto word_num_bits = sizeof(word_type) * 8;
    static constexpr auto num_bits = N;
    static constexpr auto num_words = num_bits / word_num_bits;

    static_assert(N >= 2 * word_num_bits, "Number of bits must be at lest 128");
    static_assert(N % word_num_bits == 0, "Number of bits must be a multiply of 64");

private:
    uint64_t words_[num_words];

public:
    /// Tag type for constructing without zero-initialization.
    struct uninit_tag {};

    constexpr uint() noexcept : words_{} {}

    /// Construct without zero-initializing the storage at runtime.
    /// The caller MUST fully overwrite all words before reading.
    /// In constexpr context, zero-inits for correctness; at runtime, truly uninitialized.
    constexpr explicit uint(uninit_tag) noexcept
    {
        if (std::is_constant_evaluated())
            for (auto& w : words_)
                w = 0;
    }

    /// Implicit converting constructor for any smaller uint type.
    template <unsigned M>
    constexpr explicit(false) uint(const uint<M>& x) noexcept
        requires(M < N)
    {
        for (size_t i = 0; i < uint<M>::num_words; ++i)
            words_[i] = x[i];
        for (size_t i = uint<M>::num_words; i < num_words; ++i)
            words_[i] = 0;
    }

#if INTX_HAS_BUILTIN_INT128
    constexpr explicit(false) uint(builtin_uint128 x) noexcept
      : words_{uint64_t(x), uint64_t(x >> 64)}
    {}
#endif

    template <typename... T>
    constexpr explicit(false) uint(T... v) noexcept
        requires std::conjunction_v<std::is_convertible<T, uint64_t>...>
      : words_{static_cast<uint64_t>(v)...}
    {}

    /// Constructs from words with words[0] being the least significant word.
    /// The size of the span must be less than or equal to num_words.
    constexpr explicit uint(std::span<const uint64_t> words) noexcept
    {
        INTX_REQUIRE(words.size() <= num_words);
        std::ranges::copy(words, words_);
        for (size_t i = words.size(); i < num_words; ++i)
            words_[i] = 0;
    }

    constexpr uint64_t& operator[](size_t i) noexcept { return words_[i]; }

    constexpr const uint64_t& operator[](size_t i) const noexcept { return words_[i]; }

    constexpr explicit operator bool() const noexcept
    {
#if defined(__riscv) && __riscv_xlen == 32
        // On rv32im, short-circuit on the low word for the common case.
        // Most EVM non-zero values (booleans, addresses, counters) have non-zero
        // low bits, so this avoids loading and OR-folding all 8 uint32 halves.
        // Uses uint32_t to avoid 64-bit decomposition overhead, through a named may_alias type
        // (auto would drop the attribute): plain uint32_t loads need not see uint64_t stores.
        typedef uint32_t __attribute__((may_alias)) w32;
        const w32* const w = reinterpret_cast<const w32*>(words_);
        if ((w[0] | w[1]) != 0)
            return true;
        // Fall through: check upper words only if low word is zero.
        uint32_t upper = 0;
        for (size_t i = 2; i < num_words * 2; ++i)
            upper |= w[i];
        return upper != 0;
#else
        uint64_t folded = 0;
        for (size_t i = 0; i < num_words; ++i)
            folded |= words_[i];
        return folded != 0;
#endif
    }

    /// Explicit converting operator to smaller uint types.
    template <unsigned M>
    constexpr explicit operator uint<M>() const noexcept
        requires(M < N)
    {
        uint<M> r;
        for (size_t i = 0; i < uint<M>::num_words; ++i)
            r[i] = words_[i];
        return r;
    }

    /// Explicit converting operator for all builtin integral types.
    template <typename Int>
    constexpr explicit operator Int() const noexcept
        requires(std::is_integral_v<Int>)
    {
        static_assert(sizeof(Int) <= sizeof(uint64_t));
        return static_cast<Int>(words_[0]);
    }

    constexpr uint& operator=(uint64_t v) noexcept
    {
        words_[0] = v;
        for (size_t i = 1; i < num_words; ++i)
            words_[i] = 0;
        return *this;
    }

    template <unsigned M>
    constexpr uint& operator=(const uint<M>& x) noexcept
        requires(M <= N)
    {
        for (size_t i = 0; i < uint<M>::num_words; ++i)
            words_[i] = x[i];
        for (size_t i = uint<M>::num_words; i < num_words; ++i)
            words_[i] = 0;
        return *this;
    }

    friend constexpr uint operator+(const uint& x, const uint& y) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (N == 256) {
            if (!std::is_constant_evaluated()) {
                alignas(32) uint r = x;
                alignas(32) uint b = y;
                register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&r);
                register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&b);
                register uint32_t a2 asm("x12") = 0x01;  // ADD
                asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                return r;
            }
        }
#endif
        return addc(x, y).value;
    }

    constexpr uint& operator+=(const uint& y) noexcept { return *this = *this + y; }

    constexpr uint operator-() const noexcept { return ~*this + uint{1}; }

    friend constexpr uint operator-(const uint& x, const uint& y) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (N == 256) {
            if (!std::is_constant_evaluated()) {
                alignas(32) uint r = x;
                alignas(32) uint b = y;
                register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&r);
                register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&b);
                register uint32_t a2 asm("x12") = 0x02;  // SUB
                asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                return r;
            }
        }
#endif
        return subc(x, y).value;
    }

    constexpr uint& operator-=(const uint& y) noexcept { return *this = *this - y; }

    /// Multiplication implementation using word access
    /// and discarding the high part of the result product.
    friend constexpr uint operator*(const uint& x, const uint& y) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (N == 256) {
            if (!std::is_constant_evaluated()) {
                alignas(32) uint r = x;
                alignas(32) uint b = y;
                register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&r);
                register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&b);
                register uint32_t a2 asm("x12") = 0x08;  // MUL_LOW
                asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                return r;
            }
        }
#endif
#if defined(__riscv) && __riscv_xlen == 32
        // rv32im specialization for uint256: work with native uint32_t limbs.
        // Each uint64_t word is two uint32_t halves (little-endian), giving 8 limbs
        // per uint256. We only need the low 256 bits (limbs 0..7), so we skip
        // any product where i+j >= 8. This gives 36 mul+mulhu pairs vs 64 for full.
        // Fully unrolled column-wise accumulation with a 3-word (96-bit) accumulator
        // {c2, c1, c0} to avoid overflow: max column has 8 products, each <= 2^64-2^33+1,
        // sum <= 8*(2^64) < 2^67, plus carry_in < 2^35, well within 96 bits.
        if constexpr (N == 256)
        {
            if (!std::is_constant_evaluated())
            {
                const uint32_t a0 = static_cast<uint32_t>(x[0]);
                const uint32_t a1 = static_cast<uint32_t>(x[0] >> 32);
                const uint32_t a2 = static_cast<uint32_t>(x[1]);
                const uint32_t a3 = static_cast<uint32_t>(x[1] >> 32);
                const uint32_t a4 = static_cast<uint32_t>(x[2]);
                const uint32_t a5 = static_cast<uint32_t>(x[2] >> 32);
                const uint32_t a6 = static_cast<uint32_t>(x[3]);
                const uint32_t a7 = static_cast<uint32_t>(x[3] >> 32);

                const uint32_t b0 = static_cast<uint32_t>(y[0]);
                const uint32_t b1 = static_cast<uint32_t>(y[0] >> 32);
                const uint32_t b2 = static_cast<uint32_t>(y[1]);
                const uint32_t b3 = static_cast<uint32_t>(y[1] >> 32);
                const uint32_t b4 = static_cast<uint32_t>(y[2]);
                const uint32_t b5 = static_cast<uint32_t>(y[2] >> 32);
                const uint32_t b6 = static_cast<uint32_t>(y[3]);
                const uint32_t b7 = static_cast<uint32_t>(y[3] >> 32);

                // 96-bit accumulator {c2, c1, c0} to prevent overflow.
                // MAC: acc += (uint64_t)ai * bj
                // After each column: result limb = c0, shift right by 32.
                uint32_t c0 = 0, c1 = 0, c2 = 0;
                uint64_t t;

                // Macro: accumulate one product into {c2,c1,c0}.
                // t = (uint64_t)ai * bj; c0 += lo(t); carry -> c1 += hi(t) + carry; -> c2
                #define INTX_RV32_MAC(ai, bj)                               \
                    t = static_cast<uint64_t>(ai) * (bj);                   \
                    c0 += static_cast<uint32_t>(t);                         \
                    t = static_cast<uint64_t>(c1) +                         \
                        static_cast<uint32_t>(t >> 32) + (c0 < static_cast<uint32_t>(t)); \
                    c1 = static_cast<uint32_t>(t);                          \
                    c2 += static_cast<uint32_t>(t >> 32);

                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;

                // Column 0 (1 product)
                t = static_cast<uint64_t>(a0) * b0;
                c0 = static_cast<uint32_t>(t);
                c1 = static_cast<uint32_t>(t >> 32);
                c2 = 0;
                r0 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 1 (2 products)
                INTX_RV32_MAC(a0, b1)
                INTX_RV32_MAC(a1, b0)
                r1 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 2 (3 products)
                INTX_RV32_MAC(a0, b2)
                INTX_RV32_MAC(a1, b1)
                INTX_RV32_MAC(a2, b0)
                r2 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 3 (4 products)
                INTX_RV32_MAC(a0, b3)
                INTX_RV32_MAC(a1, b2)
                INTX_RV32_MAC(a2, b1)
                INTX_RV32_MAC(a3, b0)
                r3 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 4 (5 products)
                INTX_RV32_MAC(a0, b4)
                INTX_RV32_MAC(a1, b3)
                INTX_RV32_MAC(a2, b2)
                INTX_RV32_MAC(a3, b1)
                INTX_RV32_MAC(a4, b0)
                r4 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 5 (6 products)
                INTX_RV32_MAC(a0, b5)
                INTX_RV32_MAC(a1, b4)
                INTX_RV32_MAC(a2, b3)
                INTX_RV32_MAC(a3, b2)
                INTX_RV32_MAC(a4, b1)
                INTX_RV32_MAC(a5, b0)
                r5 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 6 (7 products)
                INTX_RV32_MAC(a0, b6)
                INTX_RV32_MAC(a1, b5)
                INTX_RV32_MAC(a2, b4)
                INTX_RV32_MAC(a3, b3)
                INTX_RV32_MAC(a4, b2)
                INTX_RV32_MAC(a5, b1)
                INTX_RV32_MAC(a6, b0)
                r6 = c0; c0 = c1; c1 = c2; c2 = 0;

                // Column 7 (8 products, only low 32 bits needed — no carry-out)
                INTX_RV32_MAC(a0, b7)
                INTX_RV32_MAC(a1, b6)
                INTX_RV32_MAC(a2, b5)
                INTX_RV32_MAC(a3, b4)
                INTX_RV32_MAC(a4, b3)
                INTX_RV32_MAC(a5, b2)
                INTX_RV32_MAC(a6, b1)
                INTX_RV32_MAC(a7, b0)
                r7 = c0;

                #undef INTX_RV32_MAC

                uint<256> p;
                p[0] = static_cast<uint64_t>(r0) | (static_cast<uint64_t>(r1) << 32);
                p[1] = static_cast<uint64_t>(r2) | (static_cast<uint64_t>(r3) << 32);
                p[2] = static_cast<uint64_t>(r4) | (static_cast<uint64_t>(r5) << 32);
                p[3] = static_cast<uint64_t>(r6) | (static_cast<uint64_t>(r7) << 32);
                return p;
            }
        }
#endif

        uint<N> p;
        for (size_t j = 0; j < num_words; j++)
        {
            uint64_t k = 0;
            for (size_t i = 0; i < (num_words - j - 1); i++)
            {
                auto a = addc(p[i + j], k);
                auto t = umul(x[i], y[j]) + uint128{a.value, a.carry};
                p[i + j] = t[0];
                k = t[1];
            }
            p[num_words - 1] += x[num_words - j - 1] * y[j] + k;
        }
        return p;
    }

    constexpr uint& operator*=(const uint& y) noexcept { return *this = *this * y; }

    friend constexpr uint operator/(const uint& x, const uint& y) noexcept
    {
        return udivrem(x, y).quot;
    }

    friend constexpr uint operator%(const uint& x, const uint& y) noexcept
    {
        return udivrem(x, y).rem;
    }

    constexpr uint& operator/=(const uint& y) noexcept { return *this = *this / y; }

    constexpr uint& operator%=(const uint& y) noexcept { return *this = *this % y; }


    constexpr uint operator~() const noexcept
    {
        uint z;
        for (size_t i = 0; i < num_words; ++i)
            z[i] = ~words_[i];
        return z;
    }

    friend constexpr uint operator|(const uint& x, const uint& y) noexcept
    {
        uint z;
        for (size_t i = 0; i < num_words; ++i)
            z[i] = x[i] | y[i];
        return z;
    }

    constexpr uint& operator|=(const uint& y) noexcept { return *this = *this | y; }

    friend constexpr uint operator&(const uint& x, const uint& y) noexcept
    {
        uint z;
        for (size_t i = 0; i < num_words; ++i)
            z[i] = x[i] & y[i];
        return z;
    }

    constexpr uint& operator&=(const uint& y) noexcept { return *this = *this & y; }

    friend constexpr uint operator^(const uint& x, const uint& y) noexcept
    {
        uint z;
        for (size_t i = 0; i < num_words; ++i)
            z[i] = x[i] ^ y[i];
        return z;
    }

    constexpr uint& operator^=(const uint& y) noexcept { return *this = *this ^ y; }

    friend constexpr bool operator==(const uint& x, const uint& y) noexcept
    {
        uint64_t folded = 0;
        for (size_t i = 0; i < num_words; ++i)
            folded |= (x[i] ^ y[i]);
        return folded == 0;
    }

    friend constexpr bool operator<(const uint& x, const uint& y) noexcept
    {
        if constexpr (N == 256)
        {
            // On rv32im (no __int128), direct word-by-word comparison from MSW to LSW
            // avoids constructing uint128 temporaries and generates tighter code.
#if defined(__riscv) && __riscv_xlen == 32
            // Compare from most significant word to least significant.
            for (size_t i = num_words; i > 0; --i)
            {
                if (x[i - 1] != y[i - 1])
                    return x[i - 1] < y[i - 1];
            }
            return false;  // Equal.
#else
            auto xp = uint128{x[2], x[3]};
            auto yp = uint128{y[2], y[3]};
            if (xp == yp)
            {
                xp = uint128{x[0], x[1]};
                yp = uint128{y[0], y[1]};
            }
            return xp < yp;
#endif
        }
        else
            return subc(x, y).carry;
    }
    friend constexpr bool operator>(const uint& x, const uint& y) noexcept { return y < x; }
    friend constexpr bool operator>=(const uint& x, const uint& y) noexcept { return !(x < y); }
    friend constexpr bool operator<=(const uint& x, const uint& y) noexcept { return !(y < x); }

    friend constexpr std::strong_ordering operator<=>(const uint& x, const uint& y) noexcept
    {
        if (x == y)
            return std::strong_ordering::equal;

        return (x < y) ? std::strong_ordering::less : std::strong_ordering::greater;
    }

    friend constexpr uint operator<<(const uint& x, uint64_t shift) noexcept
    {
        if (shift >= num_bits) [[unlikely]]
            return 0;

        if constexpr (N == 256)
        {
            constexpr auto half_bits = num_bits / 2;

            const auto xlo = uint128{x[0], x[1]};

            if (shift < half_bits)
            {
                const auto lo = xlo << shift;

                const auto xhi = uint128{x[2], x[3]};

                // Find the part moved from lo to hi.
                // The shift right here can be invalid:
                // for shift == 0 => rshift == half_bits.
                // Split it into 2 valid shifts by (rshift - 1) and 1.
                const auto rshift = half_bits - shift;
                const auto lo_overflow = (xlo >> (rshift - 1)) >> 1;
                const auto hi = (xhi << shift) | lo_overflow;
                return {lo[0], lo[1], hi[0], hi[1]};
            }

            const auto hi = xlo << (shift - half_bits);
            return {0, 0, hi[0], hi[1]};
        }
        else
        {
            constexpr auto word_bits = sizeof(uint64_t) * 8;

            const auto s = shift % word_bits;
            const auto skip = static_cast<size_t>(shift / word_bits);

            uint r;
            uint64_t carry = 0;
            for (size_t i = 0; i < (num_words - skip); ++i)
            {
                r[i + skip] = (x[i] << s) | carry;
                carry = (x[i] >> (word_bits - s - 1)) >> 1;
            }
            return r;
        }
    }

    friend constexpr uint operator<<(const uint& x, std::integral auto shift) noexcept
    {
        static_assert(sizeof(shift) <= sizeof(uint64_t));
        return x << static_cast<uint64_t>(shift);
    }

    friend constexpr uint operator<<(const uint& x, const uint& shift) noexcept
    {
        // TODO: This optimisation should be handled by operator<.
        uint64_t high_words_fold = 0;
        for (size_t i = 1; i < num_words; ++i)
            high_words_fold |= shift[i];

        if (high_words_fold != 0) [[unlikely]]
            return 0;

        return x << shift[0];
    }

    friend constexpr uint operator>>(const uint& x, uint64_t shift) noexcept
    {
        if (shift >= num_bits) [[unlikely]]
            return 0;

        if constexpr (N == 256)
        {
            constexpr auto half_bits = num_bits / 2;

            const auto xhi = uint128{x[2], x[3]};

            if (shift < half_bits)
            {
                const auto hi = xhi >> shift;

                const auto xlo = uint128{x[0], x[1]};

                // Find the part moved from hi to lo.
                // The shift left here can be invalid:
                // for shift == 0 => lshift == half_bits.
                // Split it into 2 valid shifts by (lshift - 1) and 1.
                const auto lshift = half_bits - shift;
                const auto hi_overflow = (xhi << (lshift - 1)) << 1;
                const auto lo = (xlo >> shift) | hi_overflow;
                return {lo[0], lo[1], hi[0], hi[1]};
            }

            const auto lo = xhi >> (shift - half_bits);
            return {lo[0], lo[1], 0, 0};
        }
        else
        {
            constexpr auto word_bits = sizeof(uint64_t) * 8;

            const auto s = shift % word_bits;
            const auto skip = static_cast<size_t>(shift / word_bits);

            uint r;
            uint64_t carry = 0;
            for (size_t i = 0; i < (num_words - skip); ++i)
            {
                r[num_words - 1 - i - skip] = (x[num_words - 1 - i] >> s) | carry;
                carry = (x[num_words - 1 - i] << (word_bits - s - 1)) << 1;
            }
            return r;
        }
    }

    friend constexpr uint operator>>(const uint& x, std::integral auto shift) noexcept
    {
        static_assert(sizeof(shift) <= sizeof(uint64_t));
        return x >> static_cast<uint64_t>(shift);
    }

    friend constexpr uint operator>>(const uint& x, const uint& shift) noexcept
    {
        uint64_t high_words_fold = 0;
        for (size_t i = 1; i < num_words; ++i)
            high_words_fold |= shift[i];

        if (high_words_fold != 0) [[unlikely]]
            return 0;

        return x >> shift[0];
    }

    constexpr uint& operator<<=(uint shift) noexcept { return *this = *this << shift; }
    constexpr uint& operator>>=(uint shift) noexcept { return *this = *this >> shift; }
};

using uint256 = uint<256>;


/// Signed less than comparison.
///
/// Interprets the arguments as two's complement signed integers
/// and checks the "less than" relation.
template <unsigned N>
constexpr bool slt(const uint<N>& x, const uint<N>& y) noexcept
{
    constexpr auto top_word_idx = uint<N>::num_words - 1;
    const auto x_neg = static_cast<int64_t>(x[top_word_idx]) < 0;
    const auto y_neg = static_cast<int64_t>(y[top_word_idx]) < 0;
    return ((x_neg ^ y_neg) != 0) ? x_neg : x < y;
}


template <unsigned N>
constexpr std::span<uint64_t, uint<N>::num_words> as_words(uint<N>& x) noexcept
{
    return std::span<uint64_t, uint<N>::num_words>{&x[0], uint<N>::num_words};
}

template <unsigned N>
constexpr std::span<const uint64_t, uint<N>::num_words> as_words(const uint<N>& x) noexcept
{
    return std::span<const uint64_t, uint<N>::num_words>{&x[0], uint<N>::num_words};
}

template <typename T>
inline uint8_t* as_bytes(T& x) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>);  // As in bit_cast.
    return reinterpret_cast<uint8_t*>(&x);
}

template <typename T>
inline const uint8_t* as_bytes(const T& x) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>);  // As in bit_cast.
    return reinterpret_cast<const uint8_t*>(&x);
}

#if defined(__riscv) && __riscv_xlen == 32
/// rv32im specialization: full uint256 x uint256 -> uint512 multiply using
/// native uint32_t limbs. Fully unrolled column-wise schoolbook with a 96-bit
/// accumulator {c2,c1,c0} to handle carry without overflow.
/// 64 mul+mulhu pairs, all branchless.
constexpr uint<512> umul(const uint<256>& x, const uint<256>& y) noexcept
{
#if defined(AIRBENDER)
    if (!std::is_constant_evaluated()) {
        alignas(32) uint<256> lo = x;
        alignas(32) uint<256> hi = x;
        alignas(32) uint<256> b = y;
        register uintptr_t a0 asm("x10");
        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&b);
        register uint32_t a2 asm("x12");

        a0 = reinterpret_cast<uintptr_t>(&lo);
        a2 = 0x08;  // MUL_LOW
        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");

        a0 = reinterpret_cast<uintptr_t>(&hi);
        a2 = 0x10;  // MUL_HIGH
        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");

        return {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
    }
#endif
    const uint32_t a0 = static_cast<uint32_t>(x[0]);
    const uint32_t a1 = static_cast<uint32_t>(x[0] >> 32);
    const uint32_t a2 = static_cast<uint32_t>(x[1]);
    const uint32_t a3 = static_cast<uint32_t>(x[1] >> 32);
    const uint32_t a4 = static_cast<uint32_t>(x[2]);
    const uint32_t a5 = static_cast<uint32_t>(x[2] >> 32);
    const uint32_t a6 = static_cast<uint32_t>(x[3]);
    const uint32_t a7 = static_cast<uint32_t>(x[3] >> 32);

    const uint32_t b0 = static_cast<uint32_t>(y[0]);
    const uint32_t b1 = static_cast<uint32_t>(y[0] >> 32);
    const uint32_t b2 = static_cast<uint32_t>(y[1]);
    const uint32_t b3 = static_cast<uint32_t>(y[1] >> 32);
    const uint32_t b4 = static_cast<uint32_t>(y[2]);
    const uint32_t b5 = static_cast<uint32_t>(y[2] >> 32);
    const uint32_t b6 = static_cast<uint32_t>(y[3]);
    const uint32_t b7 = static_cast<uint32_t>(y[3] >> 32);

    uint32_t c0 = 0, c1 = 0, c2 = 0;
    uint64_t t;

    // MAC: accumulate one uint32_t x uint32_t product into 96-bit {c2,c1,c0}.
    #define INTX_RV32_UMUL_MAC(ai, bj)                                      \
        t = static_cast<uint64_t>(ai) * (bj);                               \
        c0 += static_cast<uint32_t>(t);                                     \
        t = static_cast<uint64_t>(c1) +                                     \
            static_cast<uint32_t>(t >> 32) + (c0 < static_cast<uint32_t>(t)); \
        c1 = static_cast<uint32_t>(t);                                      \
        c2 += static_cast<uint32_t>(t >> 32);

    uint32_t r[16];

    // Column 0
    t = static_cast<uint64_t>(a0) * b0;
    c0 = static_cast<uint32_t>(t); c1 = static_cast<uint32_t>(t >> 32); c2 = 0;
    r[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1
    INTX_RV32_UMUL_MAC(a0, b1) INTX_RV32_UMUL_MAC(a1, b0)
    r[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2
    INTX_RV32_UMUL_MAC(a0, b2) INTX_RV32_UMUL_MAC(a1, b1)
    INTX_RV32_UMUL_MAC(a2, b0)
    r[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3
    INTX_RV32_UMUL_MAC(a0, b3) INTX_RV32_UMUL_MAC(a1, b2)
    INTX_RV32_UMUL_MAC(a2, b1) INTX_RV32_UMUL_MAC(a3, b0)
    r[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4
    INTX_RV32_UMUL_MAC(a0, b4) INTX_RV32_UMUL_MAC(a1, b3)
    INTX_RV32_UMUL_MAC(a2, b2) INTX_RV32_UMUL_MAC(a3, b1)
    INTX_RV32_UMUL_MAC(a4, b0)
    r[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5
    INTX_RV32_UMUL_MAC(a0, b5) INTX_RV32_UMUL_MAC(a1, b4)
    INTX_RV32_UMUL_MAC(a2, b3) INTX_RV32_UMUL_MAC(a3, b2)
    INTX_RV32_UMUL_MAC(a4, b1) INTX_RV32_UMUL_MAC(a5, b0)
    r[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6
    INTX_RV32_UMUL_MAC(a0, b6) INTX_RV32_UMUL_MAC(a1, b5)
    INTX_RV32_UMUL_MAC(a2, b4) INTX_RV32_UMUL_MAC(a3, b3)
    INTX_RV32_UMUL_MAC(a4, b2) INTX_RV32_UMUL_MAC(a5, b1)
    INTX_RV32_UMUL_MAC(a6, b0)
    r[6] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 7
    INTX_RV32_UMUL_MAC(a0, b7) INTX_RV32_UMUL_MAC(a1, b6)
    INTX_RV32_UMUL_MAC(a2, b5) INTX_RV32_UMUL_MAC(a3, b4)
    INTX_RV32_UMUL_MAC(a4, b3) INTX_RV32_UMUL_MAC(a5, b2)
    INTX_RV32_UMUL_MAC(a6, b1) INTX_RV32_UMUL_MAC(a7, b0)
    r[7] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 8
    INTX_RV32_UMUL_MAC(a1, b7) INTX_RV32_UMUL_MAC(a2, b6)
    INTX_RV32_UMUL_MAC(a3, b5) INTX_RV32_UMUL_MAC(a4, b4)
    INTX_RV32_UMUL_MAC(a5, b3) INTX_RV32_UMUL_MAC(a6, b2)
    INTX_RV32_UMUL_MAC(a7, b1)
    r[8] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 9
    INTX_RV32_UMUL_MAC(a2, b7) INTX_RV32_UMUL_MAC(a3, b6)
    INTX_RV32_UMUL_MAC(a4, b5) INTX_RV32_UMUL_MAC(a5, b4)
    INTX_RV32_UMUL_MAC(a6, b3) INTX_RV32_UMUL_MAC(a7, b2)
    r[9] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 10
    INTX_RV32_UMUL_MAC(a3, b7) INTX_RV32_UMUL_MAC(a4, b6)
    INTX_RV32_UMUL_MAC(a5, b5) INTX_RV32_UMUL_MAC(a6, b4)
    INTX_RV32_UMUL_MAC(a7, b3)
    r[10] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 11
    INTX_RV32_UMUL_MAC(a4, b7) INTX_RV32_UMUL_MAC(a5, b6)
    INTX_RV32_UMUL_MAC(a6, b5) INTX_RV32_UMUL_MAC(a7, b4)
    r[11] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 12
    INTX_RV32_UMUL_MAC(a5, b7) INTX_RV32_UMUL_MAC(a6, b6)
    INTX_RV32_UMUL_MAC(a7, b5)
    r[12] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 13
    INTX_RV32_UMUL_MAC(a6, b7) INTX_RV32_UMUL_MAC(a7, b6)
    r[13] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 14
    INTX_RV32_UMUL_MAC(a7, b7)
    r[14] = c0; r[15] = c1;

    #undef INTX_RV32_UMUL_MAC

    uint<512> p;
    for (unsigned w = 0; w < 8; ++w)
        p[w] = static_cast<uint64_t>(r[2 * w]) |
               (static_cast<uint64_t>(r[2 * w + 1]) << 32);
    return p;
}
#endif

template <unsigned N>
constexpr uint<2 * N> umul(const uint<N>& x, const uint<N>& y) noexcept
{
    constexpr auto num_words = uint<N>::num_words;

    uint<2 * N> p;
    for (size_t j = 0; j < num_words; ++j)
    {
        uint64_t k = 0;
        for (size_t i = 0; i < num_words; ++i)
        {
            auto a = addc(p[i + j], k);
            auto t = umul(x[i], y[j]) + uint128{a.value, a.carry};
            p[i + j] = t[0];
            k = t[1];
        }
        p[j + num_words] = k;
    }
    return p;
}

template <unsigned N>
constexpr uint<N> exp(uint<N> base, uint<N> exponent) noexcept
{
    auto result = uint<N>{1};
    if (base == 2)
        return result << exponent;

    for (size_t i = bit_width(exponent); i > 0; --i)
    {
        result *= result;
        if (bit_test(exponent, i - 1))
            result *= base;
    }
    return result;
}

template <unsigned N>
constexpr bool bit_test(const uint<N>& x, size_t bit_index) noexcept
{
    const auto w = x[bit_index / uint<N>::word_num_bits];
    const auto b = bit_index % uint<N>::word_num_bits;
    return bit_test(w, b);
}

template <unsigned N>
constexpr unsigned count_significant_words(const uint<N>& x) noexcept
{
    for (size_t i = uint<N>::num_words; i > 0; --i)
    {
        if (x[i - 1] != 0)
            return static_cast<unsigned>(i);
    }
    return 0;
}

namespace internal
{
/// count_significant_bytes() of a 64-bit word by comparisons on its significant 32-bit half,
/// without a leading-zero count. 4 to 7 instructions inline on rv32.
constexpr unsigned count_significant_bytes_by_compare(uint64_t x) noexcept
{
    const auto hi = static_cast<uint32_t>(x >> 32);
    const auto v = hi != 0 ? hi : static_cast<uint32_t>(x);
    // This form keeps the comparison results as values; nested constant ternaries become
    // branches between constant loads.
    const unsigned n =
        v >= 0x10000 ? 3 + unsigned{v >= 0x1000000} : unsigned{v >= 0x100} + unsigned{v != 0};
    return hi != 0 ? 4 + n : n;
}
}  // namespace internal

constexpr unsigned count_significant_bytes(uint64_t x) noexcept
{
    // rv32im has no leading-zero count: clz() of a 64-bit word is a call to libgcc's __clzdi2,
    // around which the caller spills its live registers. The RLP encoders and EXP's gas count
    // call this on every length and value.
#if defined(__riscv) && __riscv_xlen == 32 && !defined(__riscv_zbb)
    return internal::count_significant_bytes_by_compare(x);
#else
    return (64 - clz(x) + 7) / 8;
#endif
}

template <unsigned N>
constexpr unsigned count_significant_bytes(const uint<N>& x) noexcept
{
    const auto w = count_significant_words(x);
    return (w != 0) ? count_significant_bytes(x[w - 1]) + (w - 1) * 8 : 0;
}

template <unsigned N>
constexpr unsigned clz(const uint<N>& x) noexcept
{
    constexpr unsigned num_words = uint<N>::num_words;
    const auto s = count_significant_words(x);
    if (s == 0)
        return num_words * 64;
    return clz(x[s - 1]) + (num_words - s) * 64;
}

template <unsigned N>
constexpr unsigned ctz(const uint<N>& x) noexcept
{
    for (size_t i = 0; i < uint<N>::num_words; ++i)
    {
        if (x[i] != 0)
            return static_cast<unsigned>(i * uint<N>::word_num_bits) + ctz(x[i]);
    }
    return uint<N>::num_bits;
}

/// Counts the number of bits needed to represent the value. For 0 returns 0.
template <unsigned N>
constexpr unsigned bit_width(const uint<N>& x) noexcept
{
    return uint<N>::num_bits - clz(x);
}

namespace internal
{
/// Counts the number of zero leading bits in nonzero argument x.
constexpr unsigned clz_nonzero(uint64_t x) noexcept
{
    INTX_REQUIRE(x != 0);
    return static_cast<unsigned>(std::countl_zero(x));
}


template <unsigned M, unsigned N>
struct normalized_div_args  // NOLINT(cppcoreguidelines-pro-type-member-init)
{
    uint<N> divisor;
    uint<M + 64> numerator;
    unsigned shift;
};

/// Normalizes the division arguments.
/// @param num_divisor_words  The number of significant words of the denominator.
///
/// TODO: Consider passing the denominator as std::span of its significant words.
template <unsigned M, unsigned N>
[[gnu::always_inline]] constexpr normalized_div_args<M, N> normalize(
    const uint<M>& numerator, const uint<N>& denominator, size_t num_divisor_words) noexcept
{
    const auto u = as_words(numerator);
    const auto v = as_words(denominator);

    normalized_div_args<M, N> na;
    const auto un = as_words(na.numerator);
    const auto vn = as_words(na.divisor);

    const auto n = num_divisor_words;

    na.shift = clz_nonzero(v[n - 1]);  // Use clz_nonzero() to avoid clang analyzer's warning.
    // TODO: Confirm on Mainnet data that the non-zero shift is the likely case.
    //   E.g. the secp256k1 field prime and group order have the shift 0.
    if (na.shift) [[likely]]
    {
        shl(vn, v, na.shift);  // The bits shifted out of v are zero.
        shl(un, u, na.shift);
    }
    else
    {
        na.numerator = numerator;
        na.divisor = denominator;
    }
    return na;
}

/// Divides arbitrary long unsigned integer by 64-bit unsigned integer (1 word).
/// @param u  The normalized numerator words. It will contain the quotient after execution.
/// @param d  The normalized divisor.
/// @return   The remainder.
constexpr uint64_t udivrem_by1(std::span<uint64_t> u, uint64_t d) noexcept
{
    INTX_REQUIRE(u.size() >= 2);

    const auto reciprocal = reciprocal_2by1(d);

    auto rem = u[u.size() - 1];  // Set the top word as remainder.
    u[u.size() - 1] = 0;         // Reset the word being a part of the result quotient.

    auto it = u.end() - 2;
    while (true)
    {
        std::tie(*it, rem) = udivrem_2by1({*it, rem}, d, reciprocal);
        if (it == u.begin())
            break;
        --it;
    }

    return rem;
}

/// Divides arbitrary long unsigned integer by 128-bit unsigned integer (2 words).
/// @param u  The normalized numerator words. It will contain the quotient after execution.
/// @param d  The normalized divisor.
/// @return   The remainder.
constexpr uint128 udivrem_by2(std::span<uint64_t> u, uint128 d) noexcept
{
    INTX_REQUIRE(u.size() >= 3);

    const auto reciprocal = reciprocal_3by2(d);

    auto rem = uint128{u[u.size() - 2], u[u.size() - 1]};  // Set the 2 top words as remainder.
    u[u.size() - 1] = u[u.size() - 2] = 0;  // Reset the words being a part of the result quotient.

    auto it = u.end() - 3;
    while (true)
    {
        std::tie(*it, rem) = udivrem_3by2(rem[1], rem[0], *it, d, reciprocal);
        if (it == u.begin())
            break;
        --it;
    }

    return rem;
}

/// Add y to x as: x[] += y[].
constexpr bool add(uint64_t x[], std::span<const uint64_t> y) noexcept
{
    // OPT: Add MinLen template parameter and unroll first loop iterations.
    INTX_REQUIRE(y.size() >= 2);

    bool carry = false;
    for (size_t i = 0; i < y.size(); ++i)
        std::tie(x[i], carry) = addc(x[i], y[i], carry);
    return carry;
}

/// Subtract y multiplied by multiplier from x and store the result in r
/// as: r[] = x[] - multiplier * y[].
constexpr uint64_t submul(
    uint64_t r[], const uint64_t x[], std::span<const uint64_t> y, uint64_t multiplier) noexcept
{
    // OPT: Add MinLen template parameter and unroll first loop iterations.
    INTX_REQUIRE(!y.empty());

    uint64_t borrow = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        const auto s = x[i] - borrow;
        const auto p = umul(y[i], multiplier);
        borrow = p[1] + (x[i] < s);
        r[i] = s - p[0];
        borrow += (s < r[i]);
    }
    return borrow;
}

/// Subtract y multiplied by multiplier from x as: x[] -= multiplier * y[].
constexpr uint64_t submul(uint64_t x[], std::span<const uint64_t> y, uint64_t multiplier) noexcept
{
    return submul(x, x, y, multiplier);
}

constexpr void udivrem_knuth(
    uint64_t q[], std::span<uint64_t> u, std::span<const uint64_t> d) noexcept
{
    INTX_REQUIRE(d.size() >= 3);
    INTX_REQUIRE(u.size() > d.size());

    const auto divisor = uint128{d[d.size() - 2], d[d.size() - 1]};
    const auto reciprocal = reciprocal_3by2(divisor);
    const auto dlen = d.size();
    // The top 2 words of the partial remainder.
    auto rem = uint128{u[u.size() - 2], u[u.size() - 1]};
    for (size_t j = u.size() - dlen - 1; true; --j)
    {
        const auto u0 = u[j + dlen - 2];

        uint64_t qhat{};
        if (INTX_UNLIKELY(rem == divisor))  // Division overflows.
        {
            qhat = ~uint64_t{0};

            // The borrow equals the top word u[j + dlen], which is not read again.
            submul(&u[j], d, qhat);
        }
        else
        {
            uint128 rhat;
            std::tie(qhat, rhat) = udivrem_3by2(rem[1], rem[0], u0, divisor, reciprocal);

            bool carry{};
            const auto overflow = submul(&u[j], d.subspan(0, d.size() - 2), qhat);
            std::tie(u[j + dlen - 2], carry) = subc(rhat[0], overflow);
            std::tie(u[j + dlen - 1], carry) = subc(rhat[1], carry);

            if (INTX_UNLIKELY(carry))
            {
                --qhat;
                add(&u[j], d);
            }
        }
        // The top 2 words of the next partial remainder. Read before the q[j] store
        // (q may alias u) so the compiler can reuse the values just stored.
        rem = {u[j + dlen - 2], u[j + dlen - 1]};

        q[j] = qhat;  // Store quotient digit.
        if (j == 0)   // Loop exit condition.
            break;
    }
}

/// Divides u by v when the quotient fits in a single word: v has TOP + 1 significant words
/// and u has at most one more word, which is less than v[TOP].
///
/// The quotient is estimated from the top word of the normalized divisor with 2/1 division
/// and corrected with the remainder computed on the original (not normalized) operands.
/// The word v[TOP] must not be zero.
template <size_t TOP, unsigned M, unsigned N>
constexpr div_result<uint<M>, uint<N>> udivrem_top1(const uint<M>& u, const uint<N>& v) noexcept
{
    static_assert(TOP >= 1 && TOP < uint<N>::num_words);
    static_assert(uint<M>::num_words == TOP + 1 || uint<M>::num_words == TOP + 2);

    // A numerator type wider than the divisor type comes from addmod(), where u > v.
    // For it, the u < v return is left out and u is not copied (see the remainder below).
    constexpr auto WIDER = uint<M>::num_words > uint<N>::num_words;

    uint64_t ue = 0;  // The extra word of u above the significant words of v.
    if constexpr (uint<M>::num_words == TOP + 2)
        ue = u[TOP + 1];

    if constexpr (!WIDER)
    {
        if (uint128{u[TOP], ue} < v[TOP])  // u < v.
            return {0, static_cast<uint<N>>(u)};
    }

    // The top 64 bits of the normalized divisor and the matching 2 words of the numerator.
    const auto s = clz_nonzero(v[TOP]);
    const auto vt = fshl(v[TOP], v[TOP - 1], s);
    const auto ut_hi = fshl(ue, u[TOP], s);  // Less than vt because ue < v[TOP].
    const auto ut_lo = fshl(u[TOP], u[TOP - 1], s);

    // The estimate is at most 2 too large (Knuth, TAOCP Vol. 2, 4.3.1, Theorem B).
    // TODO: This is a single 128/64 division (divq on x86-64). It is faster than the reciprocal
    //   on new CPUs (AMD Zen 3+, Intel Ice Lake+) but much slower on Intel Skylake and older.
    auto q = udivrem_2by1({ut_lo, ut_hi}, vt, reciprocal_2by1(vt)).quot;

    // The remainder u - q * v fits in the TOP + 1 words of v, so the borrow out of them
    // is ue, or ue + 1 or ue + 2 when the estimate is too large.
    // For the wider numerator the remainder is computed from u without a copy: GCC copies u
    // with 16-byte loads, which stall on the 8-byte stores of the sum in addmod().
    // The other types keep the copy: without it clang saves registers before the u < v return.
    auto r = [&] {
        if constexpr (WIDER)
            return uint<N>{};
        else
            return static_cast<uint<N>>(u);
    }();
    const auto vw = as_words(v).template first<TOP + 1>();
    uint64_t borrow = 0;
    if constexpr (WIDER)
        borrow = submul(as_words(r).data(), as_words(u).data(), vw, q) - ue;
    else
        borrow = submul(as_words(r).data(), vw, q) - ue;
    // The correction runs at most 2 times (usually 0).
    while (borrow != 0) [[unlikely]]
    {
        --q;
        borrow -= add(as_words(r).data(), vw);
    }
    if constexpr (TOP + 1 < uint<N>::num_words)
        r[TOP + 1] = 0;  // Clear the extra word copied from u.
    return {q, r};
}

/// Divides u by v when the significant words of v are one less than the words of u,
/// i.e. the quotient fits in two words.
///
/// The quotient is estimated from the top 2 words of the normalized divisor with two 3/2
/// divisions and corrected with the remainder, computed from the remainder of the 3/2 divisions
/// and the low words of the original (not normalized) operands.
/// The v must have exactly uint<M>::num_words - 1 significant words.
template <unsigned M, unsigned N>
constexpr div_result<uint<M>, uint<N>> udivrem_top2(const uint<M>& u, const uint<N>& v) noexcept
{
    constexpr auto TOP = uint<M>::num_words - 1;  // The top word index of u and the words of v.
    static_assert(TOP >= 3 && TOP <= uint<N>::num_words);

    // The top 128 bits of the normalized divisor and the matching 4 words of the numerator.
    const auto s = clz_nonzero(v[TOP - 1]);
    const uint128 vt{fshl(v[TOP - 2], v[TOP - 3], s), fshl(v[TOP - 1], v[TOP - 2], s)};
    const auto t3 = fshl(0, u[TOP], s);
    const auto t2 = fshl(u[TOP], u[TOP - 1], s);
    const auto t1 = fshl(u[TOP - 1], u[TOP - 2], s);
    const auto t0 = fshl(u[TOP - 2], u[TOP - 3], s);

    // The estimate is at most 2 too large (Knuth, TAOCP Vol. 2, 4.3.1, Theorem B, base 2^128).
    // The first division does not overflow: t3:t2 < 2^(64+s) <= 2^127 <= vt.
    const auto reciprocal = reciprocal_3by2(vt);
    const auto [q1, r1] = udivrem_3by2(t3, t2, t1, vt, reciprocal);
    const auto [q0, r0] = udivrem_3by2(r1[1], r1[0], t0, vt, reciprocal);
    auto q = uint128{q0, q1};

    // With k = 64 * (TOP - 2) - s, the position of the lowest bit of vt in v:
    // u - q * v = r0 * 2^k + u mod 2^k - q * (v mod 2^k),
    // so q is multiplied only by the low TOP - 2 words of v.
    std::array<uint64_t, TOP - 2> vl{};  // v mod 2^k.
    for (size_t i = 0; i < TOP - 2; ++i)
        vl[i] = v[i];
    vl[TOP - 3] = (vl[TOP - 3] << s) >> s;

    std::array<uint64_t, TOP> rw{};  // r0 * 2^k + u mod 2^k, less than v.
    for (size_t i = 0; i < TOP - 2; ++i)
        rw[i] = u[i];
    rw[TOP - 3] = fshr(r0[0], rw[TOP - 3] << s, s);
    rw[TOP - 2] = fshr(r0[1], r0[0], s);
    rw[TOP - 1] = r0[1] >> s;

    // r -= q * vl. The result is in (-2v, v), the borrow is its sign.
    const auto b0 = submul(rw.data(), vl, q0);
    const auto b1 = submul(&rw[1], vl, q1);
    bool borrow = false;
    std::tie(rw[TOP - 2], borrow) = subc(rw[TOP - 2], b0);
    std::tie(rw[TOP - 1], borrow) = subc(rw[TOP - 1], b1, borrow);

    // The correction runs at most 2 times (usually 0).
    const auto vw = as_words(v).template first<TOP>();
    while (borrow) [[unlikely]]
    {
        --q;
        borrow = !add(rw.data(), vw);
    }
    return {q, uint<N>{rw}};
}

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(INTX_DIV32_TEST)
/// Division on 32-bit words for rv32, where a 64-bit word multiplication is 4 multiplications and
/// a dozen carry instructions but the 32 x 32 -> 64 one is mul + mulhu and divu is native:
/// the algorithms above (Moller-Granlund reciprocals, Knuth's D) with B = 2^32.
namespace div32
{
using u32 = uint32_t;
using u64 = uint64_t;
/// A 32-bit word of a uint<N> (stored as 64-bit words): may alias them.
typedef uint32_t __attribute__((may_alias)) w32;

/// Leading zeros of each non-zero byte value.
inline constexpr auto clz_byte_table = []() noexcept {
    std::array<uint8_t, 256> table{};
    for (size_t i = 1; i < table.size(); ++i)
        table[i] = static_cast<uint8_t>(std::countl_zero(static_cast<uint8_t>(i)));
    return table;
}();

/// Leading zeros of a non-zero word (0 would give 24): 2 compare-and-shift steps bring its top
/// set bit into the top byte, and the table counts the rest in place of 3 more steps. libgcc's
/// __clzsi2 is a call, around which the caller spills its live registers.
constexpr unsigned clz_nonzero(u32 x) noexcept
{
    unsigned n = 0;
    if (x < 0x10000)
    {
        n += 16;
        x <<= 16;
    }
    if (x < 0x1000000)
    {
        n += 8;
        x <<= 8;
    }
    return n + clz_byte_table[x >> 24];
}

/// floor((B^2 - 1) / d) - B for normalized d: the 64/32 division (~d : B-1) / d, whose quotient
/// fits a word as ~d < d. Hacker's Delight divlu: two 32/16-digit steps on divu.
constexpr u32 reciprocal_2by1(u32 d) noexcept
{
    const u32 u1 = ~d;
    const u32 dh = d >> 16;
    const u32 dl = d & 0xffff;
    u32 q1 = u1 / dh;
    u32 rhat = u1 - q1 * dh;
    while (q1 >= 0x10000 || q1 * dl > ((rhat << 16) | 0xffff))
    {
        --q1;
        rhat += dh;
        if (rhat >= 0x10000)
            break;
    }
    const u32 u21 = (u1 << 16) + 0xffff - q1 * d;
    u32 q0 = u21 / dh;
    rhat = u21 - q0 * dh;
    while (q0 >= 0x10000 || q0 * dl > ((rhat << 16) | 0xffff))
    {
        --q0;
        rhat += dh;
        if (rhat >= 0x10000)
            break;
    }
    return (q1 << 16) | q0;
}

constexpr u32 reciprocal_3by2(u32 d1, u32 d0) noexcept
{
    auto v = reciprocal_2by1(d1);
    auto p = d1 * v;
    p += d0;
    if (p < d0)
    {
        --v;
        if (p >= d1)
        {
            --v;
            p -= d1;
        }
        p -= d1;
    }
    const auto t = u64{v} * d0;
    const auto t1 = static_cast<u32>(t >> 32);
    p += t1;
    if (p < t1)
    {
        --v;
        if (p >= d1)
        {
            if (p > d1 || static_cast<u32>(t) >= d0)
                --v;
        }
    }
    return v;
}

/// (u1 : u0) / d for u1 < d and v = reciprocal_2by1(d): returns the quotient, rem = remainder.
constexpr u32 udivrem_2by1(u32 u1, u32 u0, u32 d, u32 v, u32& rem) noexcept
{
    const u64 q = u64{v} * u1 + (u64{u1} << 32 | u0);
    auto q1 = static_cast<u32>(q >> 32) + 1;
    const auto q0 = static_cast<u32>(q);
    auto r = u0 - q1 * d;
    if (r > q0)
    {
        --q1;
        r += d;
    }
    if (r >= d)
    {
        ++q1;
        r -= d;
    }
    rem = r;
    return q1;
}

/// (u2 : u1 : u0) / d for (u2 : u1) < d and v = reciprocal_3by2(d): returns the quotient,
/// rem = the 2-word remainder.
constexpr u32 udivrem_3by2(u32 u2, u32 u1, u32 u0, u64 d, u32 v, u64& rem) noexcept
{
    const auto d1 = static_cast<u32>(d >> 32);
    const auto d0 = static_cast<u32>(d);
    const u64 q = u64{v} * u2 + (u64{u2} << 32 | u1);
    auto q1 = static_cast<u32>(q >> 32);
    const auto q0 = static_cast<u32>(q);
    const u32 r1 = u1 - q1 * d1;
    const u64 t = u64{d0} * q1;
    u64 r = (u64{r1} << 32 | u0) - t - d;
    ++q1;
    if (static_cast<u32>(r >> 32) >= q0)
    {
        --q1;
        r += d;
    }
    if (r >= d)
    {
        ++q1;
        r -= d;
    }
    rem = r;
    return q1;
}

/// x[0..n) -= y[0..n) * mult, returns the borrow out of the top word.
constexpr u32 submul(u32* x, const u32* y, size_t n, u32 mult) noexcept
{
    u32 borrow = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto s = x[i] - borrow;
        const auto p = u64{y[i]} * mult;
        borrow = static_cast<u32>(p >> 32) + (x[i] < s);
        x[i] = s - static_cast<u32>(p);
        borrow += (s < x[i]);
    }
    return borrow;
}

/// x[0..n) += y[0..n), returns the carry out.
constexpr u32 add(u32* x, const u32* y, size_t n) noexcept
{
    u32 carry = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto s = x[i] + y[i];
        const auto c1 = static_cast<u32>(s < x[i]);
        x[i] = s + carry;
        carry = c1 | static_cast<u32>(x[i] < s);
    }
    return carry;
}

/// Knuth's D over the normalized u[0..ulen) by d[0..dlen), dlen >= 3: q[0..ulen-dlen) gets the
/// quotient digits (not without WithQuot), u[0..dlen) the normalized remainder.
template <bool WithQuot>
[[gnu::always_inline]] constexpr void udivrem_knuth(
    w32* q, u32* u, size_t ulen, const u32* d, size_t dlen) noexcept
{
    const auto d1 = d[dlen - 1];
    const auto d0 = d[dlen - 2];
    const u64 divisor = u64{d1} << 32 | d0;
    const auto reciprocal = reciprocal_3by2(d1, d0);
    for (size_t j = ulen - dlen - 1;; --j)
    {
        const auto u2 = u[j + dlen];
        const auto u1 = u[j + dlen - 1];
        const auto u0 = u[j + dlen - 2];

        u32 qhat;
        if ((u64{u2} << 32 | u1) == divisor) [[unlikely]]  // Division overflows.
        {
            qhat = ~u32{0};
            u[j + dlen] = u2 - submul(&u[j], d, dlen, qhat);
        }
        else
        {
            u64 rhat;
            qhat = udivrem_3by2(u2, u1, u0, divisor, reciprocal, rhat);

            const auto overflow = submul(&u[j], d, dlen - 2, qhat);
            const auto rl = static_cast<u32>(rhat);
            const auto rh = static_cast<u32>(rhat >> 32);
            u[j + dlen - 2] = rl - overflow;
            const auto b1 = static_cast<u32>(rl < overflow);
            u[j + dlen - 1] = rh - b1;
            if (rh < b1) [[unlikely]]
            {
                --qhat;
                u[j + dlen - 1] += d1 + add(&u[j], d, dlen - 1);
            }
        }

        if constexpr (WithQuot)
            q[j] = qhat;
        if (j == 0)
            break;
    }
}

/// udivrem_knuth() for a divisor length known at compile time: its inner loops unroll without
/// per-word exit tests, and the divisor, copied out of d, stays in registers.
template <size_t DLEN, bool WithQuot>
[[gnu::always_inline]] constexpr void udivrem_knuth_fixed(
    w32* q, u32* u, size_t ulen, const u32* d) noexcept
{
    u32 dr[DLEN];
    for (size_t i = 0; i < DLEN; ++i)
        dr[i] = d[i];
    udivrem_knuth<WithQuot>(q, u, ulen, dr, DLEN);
}

/// r[0..n) = u[0..n) >> shift: the remainder out of the normalized one.
[[gnu::always_inline]] constexpr void unnormalize(
    w32* r, const u32* u, size_t n, unsigned shift) noexcept
{
    if (shift != 0)
    {
        u32 lo = u[0];
        for (size_t i = 0; i < n - 1; ++i)
        {
            const u32 hi = u[i + 1];
            r[i] = (lo >> shift) | (hi << (32 - shift));
            lo = hi;
        }
        r[n - 1] = lo >> shift;
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
            r[i] = u[i];
    }
}

template <typename T>
[[gnu::always_inline]] inline T uninit_uint() noexcept
{
    if constexpr (requires { typename T::uninit_tag; })
        return T{typename T::uninit_tag{}};
    else
        return T{};
}

/// The number of significant words of the W-word number w (0 for 0).
template <size_t W>
[[gnu::always_inline]] inline size_t significant_words(const w32* w) noexcept
{
    size_t n = W;
    while (n > 0 && w[n - 1] == 0)
        --n;
    return n;
}

/// x / y into qw[0..M/32) with WithQuot and x % y into rw[0..N/32) with WithRem, for x of m and
/// y of n significant words (n != 0). Either result may be y itself: y is read in full (the
/// normalized divisor is built from it) before the first store to qw or rw, and x is never a
/// result.
template <unsigned M, unsigned N, bool WithQuot, bool WithRem>
[[gnu::always_inline]] inline void divide(
    const w32* uw, size_t m, const w32* vw, w32* qw, w32* rw, size_t n) noexcept
{
    constexpr size_t UW = M / 32;
    constexpr size_t VW = N / 32;
    INTX_REQUIRE(n != 0);  // Division by 0.
    if (m < n)  // x < y: the quotient is 0, the remainder x.
    {
        if constexpr (WithQuot)
            for (size_t i = 0; i < UW; ++i)
                qw[i] = 0;
        if constexpr (WithRem)
            for (size_t i = 0; i < VW; ++i)
                rw[i] = i < UW ? uw[i] : 0;
        return;
    }

    // Normalize: shift both left until the divisor's top word has its top bit set. The whole
    // numerator is shifted, since with constant indices that is cheaper than a loop over its m
    // significant words. The divisor is shifted over its n significant words, or over all of them
    // under a wider numerator (MULMOD, whose modulus mostly has all words significant). un[m]
    // takes the numerator's shifted-out top bits; the words above it come out zero. Only
    // un[0..m] and dn[0..n) are used.
    const auto shift = clz_nonzero(vw[n - 1]);
    const size_t dwords = M > N ? VW : n;
    u32 un[UW + 1];
    u32 dn[VW];
    if (shift != 0)
    {
        const auto rs = 32 - shift;
        u32 hi = vw[dwords - 1];
        for (size_t i = dwords - 1; i != 0; --i)
        {
            const u32 lo = vw[i - 1];
            dn[i] = (hi << shift) | (lo >> rs);
            hi = lo;
        }
        dn[0] = hi << shift;
        hi = uw[UW - 1];
        un[UW] = hi >> rs;
        for (size_t i = UW - 1; i != 0; --i)
        {
            const u32 lo = uw[i - 1];
            un[i] = (hi << shift) | (lo >> rs);
            hi = lo;
        }
        un[0] = hi << shift;
    }
    else
    {
        for (size_t i = 0; i < dwords; ++i)
            dn[i] = vw[i];
        for (size_t i = 0; i < UW; ++i)
            un[i] = uw[i];
        un[UW] = 0;
    }
    // Count the normalized numerator's top word if significant.
    const size_t ulen = (un[m] != 0 || un[m - 1] >= dn[n - 1]) ? m + 1 : m;
    if (ulen <= n)  // x < y
    {
        if constexpr (WithQuot)
            for (size_t i = 0; i < UW; ++i)
                qw[i] = 0;
        if constexpr (WithRem)
            for (size_t i = 0; i < VW; ++i)
                rw[i] = i < UW ? uw[i] : 0;
        return;
    }

    // A full-width divisor under a wider numerator (MULMOD): the loops over its words get fixed
    // lengths, and the remainder fills all words.
    const bool full_divisor = M > N && n == VW;
    // Zero the quotient, one store per word (the remainder only if it is not filled), then write
    // the quotient digits [0, ulen - n) and the remainder over it.
    if constexpr (WithQuot)
        for (size_t i = 0; i < UW; ++i)
            qw[i] = 0;
    if constexpr (WithRem)
    {
        if (!full_divisor)
        {
            for (size_t i = 0; i < VW; ++i)
                rw[i] = 0;
        }
    }
    if (n == 1)
    {
        const auto d = dn[0];
        const auto v = reciprocal_2by1(d);
        u32 rem = un[ulen - 1];
        for (size_t i = ulen - 1; i-- != 0;)
        {
            const auto qhat = udivrem_2by1(rem, un[i], d, v, rem);
            if constexpr (WithQuot)
                qw[i] = qhat;
        }
        if constexpr (WithRem)
            rw[0] = rem >> shift;
    }
    else if (n == 2)
    {
        const u64 d = u64{dn[1]} << 32 | dn[0];
        const auto v = reciprocal_3by2(dn[1], dn[0]);
        u64 rem = u64{un[ulen - 1]} << 32 | un[ulen - 2];
        for (size_t i = ulen - 2; i-- != 0;)
        {
            const auto qhat =
                udivrem_3by2(static_cast<u32>(rem >> 32), static_cast<u32>(rem), un[i], d, v, rem);
            if constexpr (WithQuot)
                qw[i] = qhat;
        }
        if constexpr (WithRem)
        {
            rem >>= shift;
            rw[0] = static_cast<u32>(rem);
            rw[1] = static_cast<u32>(rem >> 32);
        }
    }
    else if (full_divisor)
    {
        udivrem_knuth_fixed<VW, WithQuot>(qw, un, ulen, dn);
        if constexpr (WithRem)
            unnormalize(rw, un, VW, shift);
    }
    else
    {
        udivrem_knuth<WithQuot>(qw, un, ulen, dn, n);
        if constexpr (WithRem)
            unnormalize(rw, un, n, shift);
    }
}

template <unsigned M, unsigned N>
constexpr div_result<uint<M>, uint<N>> udivrem(const uint<M>& x, const uint<N>& y) noexcept
{
    // Spelled out: `auto` would deduce plain uint32_t, dropping may_alias, and then GCC is free to
    // drop the result stores as dead stores to uint64_t words.
    const w32* const vw = reinterpret_cast<const w32*>(&y);
    // The quotient digits and the remainder go straight into the result. Every path returns this
    // one object, so GCC builds it in the caller's return slot (NRVO) instead of copying it there.
    div_result<uint<M>, uint<N>> res{uninit_uint<uint<M>>(), uninit_uint<uint<N>>()};
    size_t n = N / 32;
    while (n > 0 && vw[n - 1] == 0)
        --n;
    const w32* const uw = reinterpret_cast<const w32*>(&x);
    divide<M, N, true, true>(uw, significant_words<M / 32>(uw), vw,
        reinterpret_cast<w32*>(&res.quot), reinterpret_cast<w32*>(&res.rem), n);
    return res;
}

/// r = x % y, where r may be y itself. The remainder alone, in place: no quotient digits, no
/// result copy.
template <unsigned M, unsigned N>
[[gnu::noinline]] void urem(const uint<M>& x, const uint<N>& y, uint<N>& r) noexcept
{
    const w32* const vw = reinterpret_cast<const w32*>(&y);
    const size_t n = significant_words<N / 32>(vw);
    const w32* const uw = reinterpret_cast<const w32*>(&x);
    divide<M, N, false, true>(uw, significant_words<M / 32>(uw), vw, nullptr, reinterpret_cast<w32*>(&r), n);
}

/// As above for x of m significant words, which the caller has just counted.
template <unsigned M, unsigned N>
[[gnu::noinline]] void urem(const uint<M>& x, size_t m, const uint<N>& y, uint<N>& r) noexcept
{
    const w32* const vw = reinterpret_cast<const w32*>(&y);
    divide<M, N, false, true>(reinterpret_cast<const w32*>(&x), m, vw, nullptr,
        reinterpret_cast<w32*>(&r), significant_words<N / 32>(vw));
}

/// q = x / y, where q may be y itself and y has n significant words (the caller knows: it has
/// just looked at all of them).
template <unsigned N>
[[gnu::noinline]] void udiv(const uint<N>& x, const uint<N>& y, uint<N>& q, size_t n) noexcept
{
    const w32* const uw = reinterpret_cast<const w32*>(&x);
    divide<N, N, true, false>(uw, significant_words<N / 32>(uw), reinterpret_cast<const w32*>(&y),
        reinterpret_cast<w32*>(&q), nullptr, n);
}

using i64 = int64_t;

/// The NIST P-256 prime 2^256 - 2^224 + 2^192 + 2^96 - 1 (also the secp256r1 field prime), by words.
inline constexpr u32 p256_words[8] = {
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0, 0, 1, 0xFFFFFFFFu};

/// Folds the signed carry k of a sum w + k * 2^256 into w, at most twice, and subtracts p256 once if
/// w is still not below it: 2^256 = 2^224 - 2^192 - 2^96 + 1 (mod p256), and the carry that is
/// left after a fold is in {-1, 0, 1}, then 0.
[[gnu::always_inline]] inline void p256_finish(w32* r, u32 (&w)[8], i64 k) noexcept
{
    while (k != 0)
    {
        i64 acc = i64{w[0]} + k;
        w[0] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[1]};
        w[1] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[2]};
        w[2] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[3]} - k;
        w[3] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[4]};
        w[4] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[5]};
        w[5] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[6]} - k;
        w[6] = static_cast<u32>(acc);
        acc >>= 32;
        acc += i64{w[7]} + k;
        w[7] = static_cast<u32>(acc);
        k = acc >> 32;
    }
    bool ge = true;
    for (int i = 7; i >= 0; --i)
    {
        if (w[i] != p256_words[i])
        {
            ge = w[i] > p256_words[i];
            break;
        }
    }
    if (ge)
    {
        u32 borrow = 0;
        for (int i = 0; i < 8; ++i)
        {
            const u64 d = u64{w[i]} - p256_words[i] - borrow;
            w[i] = static_cast<u32>(d);
            borrow = static_cast<u32>(d >> 63);
        }
    }
    for (int i = 0; i < 8; ++i)
        r[i] = w[i];
}

/// r = t mod p256 for any 16-word t < 2^512, by the fast reduction of FIPS 186-4 D.2.3: 2^(32i)
/// for i >= 8 rewritten over the words below 2^256, as 8 column sums of signed words. The
/// remaining carry k * 2^256 has k in [-4, 6].
[[gnu::noinline]] inline void p256_reduce(w32* r, const w32* c) noexcept
{
    const i64 c8 = c[8], c9 = c[9], c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14],
              c15 = c[15];
    i64 acc;
    u32 w[8];
    acc = i64{c[0]} + c8 + c9 - c11 - c12 - c13 - c14;
    w[0] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[1]} + c9 + c10 - c12 - c13 - c14 - c15;
    w[1] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[2]} + c10 + c11 - c13 - c14 - c15;
    w[2] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[3]} + 2 * (c11 + c12) + c13 - c15 - c8 - c9;
    w[3] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[4]} + 2 * (c12 + c13) + c14 - c9 - c10;
    w[4] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[5]} + 2 * (c13 + c14) + c15 - c10 - c11;
    w[5] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[6]} + 3 * c14 + 2 * c15 + c13 - c8 - c9;
    w[6] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[7]} + 3 * c15 + c8 - c10 - c11 - c12 - c13;
    w[7] = static_cast<u32>(acc);
    p256_finish(r, w, acc >> 32);
}

/// As p256_reduce for t < 2^288 (c[9..15] == 0): t = lo + c8 * 2^256 = lo + c8 * (2^224 - 2^192 -
/// 2^96 + 1), with a carry in {-1, 0, 1}.
[[gnu::noinline]] inline void p256_reduce_short(w32* r, const w32* c) noexcept
{
    const i64 c8 = c[8];
    i64 acc;
    u32 w[8];
    acc = i64{c[0]} + c8;
    w[0] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[1]};
    w[1] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[2]};
    w[2] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[3]} - c8;
    w[3] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[4]};
    w[4] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[5]};
    w[5] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[6]} - c8;
    w[6] = static_cast<u32>(acc);
    acc >>= 32;
    acc += i64{c[7]} + c8;
    w[7] = static_cast<u32>(acc);
    p256_finish(r, w, acc >> 32);
}

/// r = t mod (2^256 - 1) for a 16-word t: t = hi * 2^256 + lo = hi + lo.
[[gnu::noinline]] inline void ones_fold(w32* r, const w32* c) noexcept
{
    u32 w[8];
    u32 carry = 0;
    for (int i = 0; i < 8; ++i)
    {
        const u64 s = u64{c[i]} + c[i + 8] + carry;
        w[i] = static_cast<u32>(s);
        carry = static_cast<u32>(s >> 32);
    }
    // hi + lo <= 2^257 - 2, so the end-around carry cannot carry again. All ones is 0.
    u32 all = 0xFFFFFFFFu;
    for (int i = 0; i < 8; ++i)
    {
        const u64 s = u64{w[i]} + carry;
        w[i] = static_cast<u32>(s);
        carry = static_cast<u32>(s >> 32);
        all &= w[i];
    }
    const u32 keep = all == 0xFFFFFFFFu ? 0 : 0xFFFFFFFFu;
    for (int i = 0; i < 8; ++i)
        r[i] = w[i] & keep;
}

/// r = t mod 2^k = the low k bits of t, for k < 256.
[[gnu::noinline]] inline void pow2_mask(w32* r, const w32* c, unsigned k) noexcept
{
    const unsigned word = k / 32, bit = k % 32;
    for (unsigned i = 0; i < 8; ++i)
    {
        u32 v = c[i];
        if (i > word || (i == word && bit == 0))
            v = 0;
        else if (i == word)
            v &= (u32{1} << bit) - 1;
        r[i] = v;
    }
}

/// The head of mulmod_reduce, defined below: the copy of a product below the modulus and the
/// closed forms of the sparse moduli.
[[gnu::always_inline]] inline bool mulmod_reduce_closed(const uint<512>& p, uint<256>& m) noexcept;

/// m = x * y % m from the 512-bit product p, into m's slot (m != 0). The moduli the corpus
/// multiplies by with a sparse form get a closed form in place of the Knuth division: the NIST
/// P-256 prime, 2^256 - 1 and powers of two. A product below m is its own remainder. Everything
/// else goes to urem with the product's length, which is scanned here once. All of m is read before
/// the first store into it, and the closed forms read only p. Every path ends in a tail call, so
/// there is no frame.
[[gnu::noinline]] inline void mulmod_reduce(const uint<512>& p, uint<256>& m) noexcept
{
    if (mulmod_reduce_closed(p, m))
        return;
    const w32* const pw = reinterpret_cast<const w32*>(&p);
    urem(p, significant_words<16>(pw), m, m);
}

/// The head of mulmod_reduce: the copy of a product below the modulus and the closed forms of the
/// sparse moduli (the NIST P-256 prime, 2^256 - 1 and powers of two). Returns false when none of
/// them applies and the caller has to reduce p itself (div32's urem, or a Montgomery reduction).
/// All of m is read before the first store into it, and the closed forms read only p.
[[gnu::always_inline]] inline bool mulmod_reduce_closed(const uint<512>& p, uint<256>& m) noexcept
{
    const w32* const pw = reinterpret_cast<const w32*>(&p);
    w32* const mw = reinterpret_cast<w32*>(&m);
    // Top down, as a chain of loads and branches: a conjunction of 8 compares compiles to loads
    // and ors on every call.
    const size_t len = significant_words<16>(pw);
    if (len <= 8)
    {
        bool below = false;
        for (int i = 7; i >= 0; --i)
        {
            const u32 a = pw[i], b = mw[i];
            if (a != b)
            {
                below = a < b;
                break;
            }
        }
        if (below)
        {
            for (size_t i = 0; i < 8; ++i)
                mw[i] = pw[i];
            return true;
        }
    }
    const u32 m7 = mw[7];
    if (m7 == 0xFFFFFFFFu)
    {
        if (mw[6] == 1 && (mw[5] | mw[4] | mw[3]) == 0 && (mw[2] & mw[1] & mw[0]) == 0xFFFFFFFFu)
        {
            if (len <= 9)
                p256_reduce_short(mw, pw);
            else
                p256_reduce(mw, pw);
            return true;
        }
        if ((mw[6] & mw[5] & mw[4] & mw[3] & mw[2] & mw[1] & mw[0]) == 0xFFFFFFFFu)
        {
            ones_fold(mw, pw);
            return true;
        }
    }
    else if ((m7 & (m7 - 1)) == 0) [[unlikely]]
    {
        // The top word is 0 or a single bit: a power of two if it is the only non-zero word, found
        // from the top (m7 first: the 2^251 + ... prime of StarkNet has a single bit there).
        unsigned t = 7;
        u32 top = m7;
        u32 low;
        if (top != 0)
            low = mw[6] | mw[5] | mw[4] | mw[3] | mw[2] | mw[1] | mw[0];
        else if ((top = mw[6]) != 0)
            t = 6, low = mw[5] | mw[4] | mw[3] | mw[2] | mw[1] | mw[0];
        else if ((top = mw[5]) != 0)
            t = 5, low = mw[4] | mw[3] | mw[2] | mw[1] | mw[0];
        else if ((top = mw[4]) != 0)
            t = 4, low = mw[3] | mw[2] | mw[1] | mw[0];
        else if ((top = mw[3]) != 0)
            t = 3, low = mw[2] | mw[1] | mw[0];
        else if ((top = mw[2]) != 0)
            t = 2, low = mw[1] | mw[0];
        else if ((top = mw[1]) != 0)
            t = 1, low = mw[0];
        else
            t = 0, top = mw[0], low = 0;
        if ((top & (top - 1)) == 0 && low == 0)
        {
            pow2_mask(mw, pw, 32 * t + 31 - clz_nonzero(top));
            return true;
        }
    }
    return false;
}
}  // namespace div32
#endif

}  // namespace internal

template <unsigned M, unsigned N>
constexpr div_result<uint<M>, uint<N>> udivrem(const uint<M>& u, const uint<N>& v) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    if (!std::is_constant_evaluated())
        return internal::div32::udivrem(u, v);
#endif
    // For equal sizes, a divisor with the top word set gives a single word quotient.
    constexpr auto TOP1 = M == N;
    // A divisor with one significant word less than the numerator gives a two word quotient.
    // It needs at least 3 divisor words to form the normalized top 128 bits.
    constexpr auto TOP2 =
        (uint<M>::num_words >= 4) && (uint<M>::num_words - 1 == uint<N>::num_words - size_t{TOP1});

    size_t num_divisor_words = uint<N>::num_words;
    if constexpr (TOP1)
    {
        if (v[num_divisor_words - 1] != 0)
            return internal::udivrem_top1<uint<N>::num_words - 1>(u, v);
        --num_divisor_words;  // Skip the zero top word.
    }
    if constexpr (TOP2)
    {
        constexpr auto TOP = uint<M>::num_words - 2;  // The top word index of v.
        if (v[TOP] != 0)
        {
            // The quotient fits in a single word if the top word of u is less than v[TOP].
            if (u[TOP + 1] < v[TOP])
                return internal::udivrem_top1<TOP>(u, v);
            return internal::udivrem_top2(u, v);
        }
        --num_divisor_words;  // Skip the zero word.
    }
    while (num_divisor_words > 1 && v[num_divisor_words - 1] == 0)  // The divisor is not zero.
        --num_divisor_words;
    const auto n = num_divisor_words;

    // The number of significant words of the numerator, at least 1.
    size_t m = uint<M>::num_words;
    while (m > 1 && u[m - 1] == 0)
        --m;

    // The numerator words to divide: the m words and the word with the bits shifted out of the
    // top by the normalization. That word is below the top divisor word, so the division can
    // start from it. It is skipped if the top word of u is below the top word of v: then the
    // shifted out bits are 0 and so is the top quotient word. No quotient words means u < v.
    const auto num_numerator_words = m + size_t{u[m - 1] >= v[n - 1]};
    if (num_numerator_words <= n)
        return {0, static_cast<uint<N>>(u)};

    auto na = internal::normalize(u, v, n);

    // The span of the normalized numerator words to divide (modified in place).
    const auto un = as_words(na.numerator).subspan(0, num_numerator_words);
    // The span of the normalized divisor significant words.
    const auto dn = as_words(na.divisor).subspan(0, n);

    INTX_REQUIRE(!dn.empty());
    INTX_REQUIRE(dn.size() <= uint<N>::num_words);
    INTX_REQUIRE(un.size() <= uint<M>::num_words + 1);

    static_assert(uint<N>::num_words >= 2, "no support for uint<64> yet");
    if (dn.size() == 1)
    {
        const auto r = internal::udivrem_by1(un, dn[0]);
        return {static_cast<uint<M>>(na.numerator), r >> na.shift};
    }

    // The max number of divisor words here: the divisors with more words are handled above.
    constexpr auto MAX_DIVISOR_WORDS = uint<N>::num_words - size_t{TOP1} - size_t{TOP2};
    // The Knuth division handles divisors of at least 3 words.
    constexpr auto KNUTH_REACHABLE = MAX_DIVISOR_WORDS >= 3;
    if (!KNUTH_REACHABLE || dn.size() == 2)
    {
        const auto r = internal::udivrem_by2(un, static_cast<uint128>(na.divisor));
        return {static_cast<uint<M>>(na.numerator), r >> na.shift};
    }

    if constexpr (KNUTH_REACHABLE)
    {
        uint<M> q;
        internal::udivrem_knuth(&q[0], un, dn);

        uint<N> r;
        shr(as_words(r).first(dn.size()), un.first(dn.size()), na.shift);

        return {q, r};
    }
}

template <unsigned N>
constexpr div_result<uint<N>> sdivrem(const uint<N>& u, const uint<N>& v) noexcept
{
    const auto sign_mask = uint<N>{1} << (uint<N>::num_bits - 1);
    auto u_is_neg = (u & sign_mask) != 0;
    auto v_is_neg = (v & sign_mask) != 0;

    auto u_abs = u_is_neg ? -u : u;
    auto v_abs = v_is_neg ? -v : v;

    auto q_is_neg = u_is_neg ^ v_is_neg;

    auto res = udivrem(u_abs, v_abs);

    return {q_is_neg ? -res.quot : res.quot, u_is_neg ? -res.rem : res.rem};
}

constexpr uint256 bswap(const uint256& x) noexcept
{
#if defined(__riscv) && __riscv_xlen == 32 && !defined(__riscv_zbb)
    // Inline bswap32 logic with shared mask constants across all 8 half-word swaps.
    // Saves ~30 insns vs 8 separate bswap32 calls that each reconstruct the masks.
    if (!std::is_constant_evaluated())
    {
        // Hoist the byte-swap mask constants so they're computed once.
        const uint32_t m1 = 0xFF00FF00u;
        const uint32_t m2 = 0x00FF00FFu;
        auto bs = [m1, m2](uint32_t v) -> uint32_t {
            const auto a = ((v << 8) & m1) | ((v >> 8) & m2);
            return (a << 16) | (a >> 16);
        };
        // bswap64(w) = (bswap32(lo) << 32) | bswap32(hi), then reverse word order.
        auto bs64 = [&bs](uint64_t w) -> uint64_t {
            const auto lo = static_cast<uint32_t>(w);
            const auto hi = static_cast<uint32_t>(w >> 32);
            return (static_cast<uint64_t>(bs(lo)) << 32) | bs(hi);
        };
        return {bs64(x[3]), bs64(x[2]), bs64(x[1]), bs64(x[0])};
    }
#endif
    return {bswap(x[3]), bswap(x[2]), bswap(x[1]), bswap(x[0])};
}

template <unsigned N>
constexpr uint<N> bswap(const uint<N>& x) noexcept
{
    constexpr auto num_words = uint<N>::num_words;
    uint<N> z;
    for (size_t i = 0; i < num_words; ++i)
        z[num_words - 1 - i] = bswap(x[i]);
    return z;
}


constexpr uint256 addmod(const uint256& x, const uint256& y, const uint256& mod) noexcept
{
    // Fast path for mod >= 2^192, with x and y at most slightly bigger than mod.
    // This is always the case when x and y are already reduced modulo mod.
    // Based on https://github.com/holiman/uint256/pull/86.
    if ((mod[3] != 0) && (x[3] <= mod[3]) && (y[3] <= mod[3]))
    {
        // Normalize x in case it is bigger than mod.
        auto xn = x;
        auto xd = subc(x, mod);
        if (!xd.carry)
            xn = xd.value;

        // Normalize y in case it is bigger than mod.
        auto yn = y;
        auto yd = subc(y, mod);
        if (!yd.carry)
            yn = yd.value;

        auto a = addc(xn, yn);
        auto av = a.value;
        auto b = subc(av, mod);
        auto bv = b.value;
        if (a.carry || !b.carry)
            return bv;
        return av;
    }

    auto s = addc(x, y);
    uint<256 + 64> n = s.value;
    n[4] = s.carry;
    return udivrem(n, mod).rem;
}

constexpr uint256 mulmod(const uint256& x, const uint256& y, const uint256& mod) noexcept
{
    return udivrem(umul(x, y), mod).rem;
}

#define INTX_JOIN(X, Y) X##Y
/// Define type alias uintN = uint<N> and the matching literal ""_uN.
/// The literal operators are defined in the intx::literals namespace.
#define DEFINE_ALIAS_AND_LITERAL(N)                               \
    using uint##N = uint<N>;                                      \
    namespace literals                                            \
    {                                                             \
    consteval uint##N INTX_JOIN(operator"", _u##N)(const char* s) \
    {                                                             \
        return from_string<uint##N>(s);                           \
    }                                                             \
    }
DEFINE_ALIAS_AND_LITERAL(128)
DEFINE_ALIAS_AND_LITERAL(192)
DEFINE_ALIAS_AND_LITERAL(256)
DEFINE_ALIAS_AND_LITERAL(320)
DEFINE_ALIAS_AND_LITERAL(384)
DEFINE_ALIAS_AND_LITERAL(448)
DEFINE_ALIAS_AND_LITERAL(512)
#undef DEFINE_ALIAS_AND_LITERAL
#undef INTX_JOIN

using namespace literals;

/// Convert native representation to/from little-endian byte order.
/// intx and built-in integral types are supported.
template <typename T>
constexpr T to_little_endian(const T& x) noexcept
{
    if constexpr (std::endian::native == std::endian::little)
        return x;
    else if constexpr (std::is_integral_v<T>)
        return bswap(x);
    else  // Wordwise bswap.
    {
        T r;
        for (size_t i = 0; i < T::num_words; ++i)
            r[i] = bswap(x[i]);
        return r;
    }
}

/// Convert native representation to/from big-endian byte order.
/// intx and built-in integral types are supported.
template <typename T>
constexpr T to_big_endian(const T& x) noexcept
{
    if constexpr (std::endian::native == std::endian::little)
        return bswap(x);
    else if constexpr (std::is_integral_v<T>)
        return x;
    else  // Swap words.
    {
        T r;
        for (size_t i = 0; i < T::num_words; ++i)
            r[T::num_words - 1 - i] = x[i];
        return r;
    }
}

namespace le  // Conversions to/from LE bytes.
{
template <typename T, unsigned M>
inline T load(const uint8_t (&src)[M]) noexcept
{
    static_assert(
        M == sizeof(T), "the size of source bytes must match the size of the destination uint");
    T x;
    std::memcpy(&x, src, sizeof(x));
    return to_little_endian(x);
}

template <typename T>
inline void store(uint8_t (&dst)[sizeof(T)], const T& x) noexcept
{
    const auto d = to_little_endian(x);
    std::memcpy(dst, &d, sizeof(d));
}

namespace unsafe
{
template <typename T>
inline T load(const uint8_t* src) noexcept
{
    T x;
    std::memcpy(&x, src, sizeof(x));
    return to_little_endian(x);
}

template <typename T>
inline void store(uint8_t* dst, const T& x) noexcept
{
    const auto d = to_little_endian(x);
    std::memcpy(dst, &d, sizeof(d));
}
}  // namespace unsafe
}  // namespace le


#if defined(__riscv) && !defined(__riscv_zbb)
namespace internal
{
/// Writes the 32 bytes at s in reverse byte order to d: d[k] = bswap32(s[7 - k]) word-wise.
/// Both must be 4-byte aligned; d == s is allowed (all words are loaded first).
///
/// This is the whole 256-bit big-endian conversion in 8 loads, 8 stores and the shared-mask
/// swaps, instead of copying into an aligned temporary and swapping that (8 more loads and
/// stores), or swapping 64-bit words as pairs of 32-bit halves.
[[gnu::always_inline]] inline void bswap256_words(void* d, const void* s) noexcept
{
    // Words of a named may_alias type (auto would drop the attribute): d is a uint256, whose
    // words are uint64_t, and s a byte buffer. Through plain uint32_t the stores to d do not
    // alias the uint64_t reads of the result, and GCC is free to drop them as dead stores.
    typedef uint32_t __attribute__((may_alias)) w32;
    const w32* const sw = static_cast<const w32*>(s);
    w32* const dw = static_cast<w32*>(d);
    uint32_t w[8];
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
        w[i] = sw[i];
    const uint32_t m1 = 0xFF00FF00u;
    const uint32_t m2 = 0x00FF00FFu;
#pragma GCC unroll 8
    for (int k = 0; k < 8; ++k)
    {
        const auto v = w[7 - k];
        const auto a = ((v << 8) & m1) | ((v >> 8) & m2);
        dw[k] = (a << 16) | (a >> 16);
    }
}

[[gnu::always_inline]] inline bool is_aligned4(const void* p) noexcept
{
    return (reinterpret_cast<uintptr_t>(p) & 3) == 0;
}

/// Writes the 32 bytes at s in reverse byte order to d with byte loads and stores.
/// d and s must not overlap; neither needs any alignment.
///
/// Without rev8, a word swap costs 8 ALU instructions, so lw + swap + sw is 10 instructions
/// per 4 bytes (plus the masks) where lbu + sb is 8, and memory need not be word-aligned.
/// Use it when the destination is memory anyway; a value built in registers is better served
/// by bswap256_words. One asm block keeps it to one scratch register: as separate C byte
/// copies GCC hoists the loads and spills hot registers around them.
[[gnu::always_inline]] inline void bswap256_bytes(void* d, const void* s) noexcept
{
    using Bytes = uint8_t[32];
    uint32_t t;
#define INTX_RB(si, di) "lbu %[t], " #si "(%[s])\n\tsb %[t], " #di "(%[d])\n\t"
    asm(
        INTX_RB(31, 0) INTX_RB(30, 1) INTX_RB(29, 2) INTX_RB(28, 3)
        INTX_RB(27, 4) INTX_RB(26, 5) INTX_RB(25, 6) INTX_RB(24, 7)
        INTX_RB(23, 8) INTX_RB(22, 9) INTX_RB(21, 10) INTX_RB(20, 11)
        INTX_RB(19, 12) INTX_RB(18, 13) INTX_RB(17, 14) INTX_RB(16, 15)
        INTX_RB(15, 16) INTX_RB(14, 17) INTX_RB(13, 18) INTX_RB(12, 19)
        INTX_RB(11, 20) INTX_RB(10, 21) INTX_RB(9, 22) INTX_RB(8, 23)
        INTX_RB(7, 24) INTX_RB(6, 25) INTX_RB(5, 26) INTX_RB(4, 27)
        INTX_RB(3, 28) INTX_RB(2, 29) INTX_RB(1, 30) INTX_RB(0, 31)
        : [t] "=&r"(t), "=m"(*static_cast<Bytes*>(d))
        : [d] "r"(d), [s] "r"(s), "m"(*static_cast<const Bytes*>(s)));
#undef INTX_RB
}

// The word-wise variants of bswap256_bytes for a 4-byte aligned big-endian side. They test the
// words from the most significant one down and store each leading zero word as one zero word;
// the first non-zero word and all after it are reversed with 3 shifts and 4 byte stores. That is
// 3 instructions per leading zero word and 8 per other word, 25 + 5 k for k significant words
// instead of 64: EVM memory words are mostly small numbers (k = 1) or addresses (k = 5).
// INTX_LZ: word n from source offset so, a zero word at destination offset dw;
// INTX_RW: word n reversed into destination bytes d0..d3; INTX_LW: the next word.
#define INTX_LZ(n, so, dw) \
    "lw %[t], " #so "(%[s])\n\tbnez %[t], 7" #n "f\n\tsw zero, " #dw "(%[d])\n\t"
#define INTX_RW(n, d0, d1, d2, d3)                                            \
    "7" #n ":\n\t"                                                            \
    "srli %[u], %[t], 24\n\tsb %[u], " #d0 "(%[d])\n\t"                       \
    "srli %[u], %[t], 16\n\tsb %[u], " #d1 "(%[d])\n\t"                       \
    "srli %[u], %[t], 8\n\tsb %[u], " #d2 "(%[d])\n\t"                        \
    "sb %[t], " #d3 "(%[d])\n\t"
#define INTX_LW(so) "lw %[t], " #so "(%[s])\n\t"

/// bswap256_bytes from 32 big-endian bytes at a 4-byte aligned s. d and s must not overlap.
[[gnu::always_inline]] inline void bswap256_from_aligned(void* d, const void* s) noexcept
{
    using Bytes = uint8_t[32];
    uint32_t t, u;
    asm(INTX_LZ(0, 0, 28) INTX_LZ(1, 4, 24) INTX_LZ(2, 8, 20) INTX_LZ(3, 12, 16)
        INTX_LZ(4, 16, 12) INTX_LZ(5, 20, 8) INTX_LZ(6, 24, 4) INTX_LZ(7, 28, 0)
        "j 79f\n\t"
        INTX_RW(0, 28, 29, 30, 31) INTX_LW(4) INTX_RW(1, 24, 25, 26, 27)
        INTX_LW(8) INTX_RW(2, 20, 21, 22, 23) INTX_LW(12) INTX_RW(3, 16, 17, 18, 19)
        INTX_LW(16) INTX_RW(4, 12, 13, 14, 15) INTX_LW(20) INTX_RW(5, 8, 9, 10, 11)
        INTX_LW(24) INTX_RW(6, 4, 5, 6, 7) INTX_LW(28) INTX_RW(7, 0, 1, 2, 3)
        "79:"
        : [t] "=&r"(t), [u] "=&r"(u), "=m"(*static_cast<Bytes*>(d))
        : [d] "r"(d), [s] "r"(s), "m"(*static_cast<const Bytes*>(s)));
}

/// bswap256_bytes to 32 big-endian bytes at a 4-byte aligned d. s is a value (its words aligned);
/// d and s must not overlap.
[[gnu::always_inline]] inline void bswap256_to_aligned(void* d, const void* s) noexcept
{
    using Bytes = uint8_t[32];
    uint32_t t, u;
    asm(INTX_LZ(0, 28, 0) INTX_LZ(1, 24, 4) INTX_LZ(2, 20, 8) INTX_LZ(3, 16, 12)
        INTX_LZ(4, 12, 16) INTX_LZ(5, 8, 20) INTX_LZ(6, 4, 24) INTX_LZ(7, 0, 28)
        "j 79f\n\t"
        INTX_RW(0, 0, 1, 2, 3) INTX_LW(24) INTX_RW(1, 4, 5, 6, 7)
        INTX_LW(20) INTX_RW(2, 8, 9, 10, 11) INTX_LW(16) INTX_RW(3, 12, 13, 14, 15)
        INTX_LW(12) INTX_RW(4, 16, 17, 18, 19) INTX_LW(8) INTX_RW(5, 20, 21, 22, 23)
        INTX_LW(4) INTX_RW(6, 24, 25, 26, 27) INTX_LW(0) INTX_RW(7, 28, 29, 30, 31)
        "79:"
        : [t] "=&r"(t), [u] "=&r"(u), "=m"(*static_cast<Bytes*>(d))
        : [d] "r"(d), [s] "r"(s), "m"(*static_cast<const Bytes*>(s)));
}
#undef INTX_LZ
#undef INTX_RW
#undef INTX_LW
}  // namespace internal
#endif

namespace be  // Conversions to/from BE bytes.
{
/// Loads an integer value from bytes of big-endian order.
/// If the size of bytes is smaller than the result, the value is zero-extended.
template <typename T, unsigned M>
inline T load(const uint8_t (&src)[M]) noexcept
{
    static_assert(M <= sizeof(T),
        "the size of source bytes must not exceed the size of the destination uint");
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    if constexpr (M == sizeof(T) && sizeof(T) == 32)
    {
        // Full-size load: byte-reverse straight from the source words into the result.
        T x{typename T::uninit_tag{}};
        if (internal::is_aligned4(src))
            internal::bswap256_words(&x, src);
        else
        {
            alignas(32) char raw_[sizeof(T)];
            std::memcpy(raw_, src, M);
            internal::bswap256_words(&x, raw_);
        }
        return x;
    }
    else
#endif
    {
        T x{};
        std::memcpy(&as_bytes(x)[sizeof(T) - M], src, M);
        x = to_big_endian(x);
        return x;
    }
}

/// Loads an integer value from the span of bytes of big-endian order.
/// If the size of bytes is smaller than the result, the value is zero-extended.
template <typename T, std::size_t Extent>
inline T load(std::span<const uint8_t, Extent> src) noexcept
{
    if constexpr (Extent != std::dynamic_extent)  // NOLINTNEXTLINE(bugprone-sizeof-expression)
        static_assert(Extent <= sizeof(T), "source bytes must not exceed the value size");
    else
        assert(src.size() <= sizeof(T));  // source bytes must not exceed the value size
    T x{};
    std::memcpy(&as_bytes(x)[sizeof(T) - src.size()], src.data(), src.size());
    x = to_big_endian(x);
    return x;
}

template <typename IntT, typename T>
inline IntT load(const T& t) noexcept
{
    return load<IntT>(t.bytes);
}

/// Stores an integer value in a bytes array in big-endian order.
template <typename T>
inline void store(uint8_t (&dst)[sizeof(T)], const T& x) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    if constexpr (sizeof(T) == 32)
        internal::bswap256_bytes(dst, &x);
    else
#endif
    {
        const auto d = to_big_endian(x);
        std::memcpy(dst, &d, sizeof(d));
    }
}

/// Stores an integer value into the span of bytes in big-endian order.
template <typename T>
inline void store(std::span<uint8_t, sizeof(T)> dst, const T& x) noexcept
{
    const auto d = to_big_endian(x);
    std::memcpy(dst.data(), &d, sizeof(d));
}

/// Stores an SrcT value in .bytes field of type DstT. The .bytes must be an array of uint8_t
/// of the size matching the size of uint.
template <typename DstT, typename SrcT>
inline DstT store(const SrcT& x) noexcept
{
    DstT r{};
    store(r.bytes, x);
    return r;
}

/// Stores the truncated value of an uint in a bytes array.
/// Only the least significant bytes from big-endian representation of the uint
/// are stored in the result bytes array up to array's size.
template <unsigned M, unsigned N>
inline void trunc(uint8_t (&dst)[M], const uint<N>& x) noexcept
{
    static_assert(M < N / 8, "destination must be smaller than the source value");
    const auto d = to_big_endian(x);
    std::memcpy(dst, &as_bytes(d)[sizeof(d) - M], M);
}

/// Stores the truncated value of an integer into the span of bytes.
/// Only the least significant bytes from big-endian representation of the uint
/// are stored in the result up to the span size.
template <unsigned N, std::size_t Extent>
inline void trunc(std::span<uint8_t, Extent> dst, const uint<N>& x) noexcept
{
    if constexpr (Extent != std::dynamic_extent)  // NOLINTNEXTLINE(bugprone-sizeof-expression)
        static_assert(Extent < sizeof(x), "destination must be smaller than the source value");
    else
        assert(dst.size() <= sizeof(x));  // destination must be not larger than the source value

    const auto d = to_big_endian(x);
    std::copy_n(&as_bytes(d)[sizeof(d) - dst.size()], dst.size(), dst.begin());
}

/// Stores the truncated value of an uint in the .bytes field of an object of type T.
template <typename T, unsigned N>
inline T trunc(const uint<N>& x) noexcept
{
    T r{};
    trunc(r.bytes, x);
    return r;
}

namespace unsafe
{

#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
/// Inline 32-byte copy: avoids calling memcpy (which is a function call under -fno-builtin)
/// for the hot 32-byte load path (MLOAD, SLOAD key conversion, etc.).
/// Uses 4-byte word loads when src is 4-byte aligned; falls back to byte loads otherwise.
inline void copy32(void* dst, const uint8_t* src) noexcept
{
    // The words of a named may_alias type, as in internal::bswap256_words: dst is mostly a uint256.
    typedef uint32_t __attribute__((may_alias)) w32;
    w32* const d = static_cast<w32*>(dst);
    if ((reinterpret_cast<uintptr_t>(src) & 3) == 0)  // 4-byte aligned
    {
        const w32* const s = reinterpret_cast<const w32*>(src);
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        d[4] = s[4]; d[5] = s[5]; d[6] = s[6]; d[7] = s[7];
    }
    else
    {
        std::memcpy(dst, src, 32);
    }
}
#endif

/// Loads an uint value from a buffer. The user must make sure
/// that the provided buffer is big enough. Therefore, marked "unsafe".
template <typename IntT>
inline IntT load(const uint8_t* src) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    if constexpr (sizeof(IntT) == 32)
    {
        IntT x{typename IntT::uninit_tag{}};
        if (internal::is_aligned4(src))
            internal::bswap256_words(&x, src);
        else
        {
            alignas(32) std::byte aligned_storage[32];
            std::memcpy(&aligned_storage, src, 32);
            internal::bswap256_words(&x, &aligned_storage);
        }
        return x;
    }
    else if constexpr (sizeof(IntT) == 8)
    {
        // Inline 8-byte BE load: construct numeric value directly from bytes.
        // Avoids memcpy function call on rv32 with -fno-builtin.
        const auto hi = static_cast<uint32_t>(src[0]) << 24
                      | static_cast<uint32_t>(src[1]) << 16
                      | static_cast<uint32_t>(src[2]) << 8
                      | static_cast<uint32_t>(src[3]);
        const auto lo = static_cast<uint32_t>(src[4]) << 24
                      | static_cast<uint32_t>(src[5]) << 16
                      | static_cast<uint32_t>(src[6]) << 8
                      | static_cast<uint32_t>(src[7]);
        return static_cast<IntT>((static_cast<uint64_t>(hi) << 32) | lo);
    }
    else if constexpr (sizeof(IntT) == 4)
    {
        // Inline 4-byte BE load: construct value directly from bytes.
        return static_cast<IntT>(
            static_cast<uint32_t>(src[0]) << 24
          | static_cast<uint32_t>(src[1]) << 16
          | static_cast<uint32_t>(src[2]) << 8
          | static_cast<uint32_t>(src[3]));
    }
    else if constexpr (sizeof(IntT) == 2)
    {
        // Inline 2-byte BE load.
        return static_cast<IntT>(
            static_cast<uint16_t>(src[0]) << 8
          | static_cast<uint16_t>(src[1]));
    }
    else
#endif
    {
        // Align bytes.
        // TODO: Using memcpy() directly triggers this optimization bug in GCC:
        //   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107837
        alignas(IntT) std::byte aligned_storage[sizeof(IntT)];
        std::memcpy(&aligned_storage, src, sizeof(IntT));
        // TODO(C++23): Use std::start_lifetime_as<uint256>().
        return to_big_endian(*reinterpret_cast<const IntT*>(&aligned_storage));
    }
}

#if defined(__riscv) && !defined(__riscv_zbb)
/// Loads 32 big-endian bytes from src into the existing value x, which must not overlap src.
/// Reversing the bytes straight into x beats building the value in registers when x lives in
/// memory anyway (an EVM stack slot), see internal::bswap256_bytes. A word-aligned src (most
/// MLOAD offsets and calldata arguments) skips the leading zero words.
inline void load_into(uint256& x, const uint8_t* src) noexcept
{
    if (internal::is_aligned4(src)) [[likely]]
        internal::bswap256_from_aligned(&x, src);
    else
        internal::bswap256_bytes(&x, src);
}

/// load_into() for a src that is known to be 4-byte aligned (a local, not memory of the EVM or
/// the witness): the leading zero words are skipped without the alignment test. src must not
/// overlap x.
inline void load_aligned_into(uint256& x, const uint8_t* src) noexcept
{
    internal::bswap256_from_aligned(&x, src);
}

/// store() for a dst that is known to be 4-byte aligned, see load_aligned_into(). dst must not
/// overlap x.
inline void store_aligned(uint8_t* dst, const uint256& x) noexcept
{
    internal::bswap256_to_aligned(dst, &x);
}
#endif

/// Stores an integer value at the provided pointer in big-endian order. The user must make sure
/// that the provided buffer is big enough to fit the value. Therefore, marked "unsafe".
template <typename T>
inline void store(uint8_t* dst, const T& x) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    if constexpr (sizeof(T) == 8)
    {
        // Inline 8-byte BE store: decompose numeric value directly into bytes.
        const auto v = static_cast<uint64_t>(x);
        const auto hi = static_cast<uint32_t>(v >> 32);
        const auto lo = static_cast<uint32_t>(v);
        dst[0] = static_cast<uint8_t>(hi >> 24);
        dst[1] = static_cast<uint8_t>(hi >> 16);
        dst[2] = static_cast<uint8_t>(hi >> 8);
        dst[3] = static_cast<uint8_t>(hi);
        dst[4] = static_cast<uint8_t>(lo >> 24);
        dst[5] = static_cast<uint8_t>(lo >> 16);
        dst[6] = static_cast<uint8_t>(lo >> 8);
        dst[7] = static_cast<uint8_t>(lo);
    }
    else if constexpr (sizeof(T) == 4)
    {
        const auto v = static_cast<uint32_t>(x);
        dst[0] = static_cast<uint8_t>(v >> 24);
        dst[1] = static_cast<uint8_t>(v >> 16);
        dst[2] = static_cast<uint8_t>(v >> 8);
        dst[3] = static_cast<uint8_t>(v);
    }
    else if constexpr (sizeof(T) == 2)
    {
        const auto v = static_cast<uint16_t>(x);
        dst[0] = static_cast<uint8_t>(v >> 8);
        dst[1] = static_cast<uint8_t>(v);
    }
    else
    {
        const auto d = to_big_endian(x);
        std::memcpy(dst, &d, sizeof(d));
    }
#else
    const auto d = to_big_endian(x);
    std::memcpy(dst, &d, sizeof(d));
#endif
}

/// Specialization for uint256.
inline void store(uint8_t* dst, const uint256& x) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // The destination is memory, so reverse the bytes straight into it (any alignment). A
    // word-aligned one (most MSTORE offsets) skips the leading zero words.
    if (internal::is_aligned4(dst)) [[likely]]
        internal::bswap256_to_aligned(dst, &x);
    else
        internal::bswap256_bytes(dst, &x);
#else
    // Store byte-swapped words in primitive temporaries. This helps with memory aliasing
    // and GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107837
    // TODO: Use std::byte instead of uint8_t.
    const auto v0 = to_big_endian(x[0]);
    const auto v1 = to_big_endian(x[1]);
    const auto v2 = to_big_endian(x[2]);
    const auto v3 = to_big_endian(x[3]);

    // Store words in reverse (big-endian) order, write addresses are ascending.
    std::memcpy(dst, &v3, sizeof(v3));
    std::memcpy(dst + 8, &v2, sizeof(v2));
    std::memcpy(dst + 16, &v1, sizeof(v1));
    std::memcpy(dst + 24, &v0, sizeof(v0));
#endif
}

}  // namespace unsafe

}  // namespace be

}  // namespace intx

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
