// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2023 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmmax/evmmax.hpp>
#include <optional>
#include <span>

namespace evmmax::ecc
{
template <int N>
struct Constant : std::integral_constant<int, N>
{
    consteval explicit(false) Constant(int v) noexcept
    {
        if (N != v)
            intx::unreachable();
    }
};
using zero_t = Constant<0>;
using one_t = Constant<1>;

/// The order specification (prime number) for a finite field.
template <typename T>
concept FieldSpec = requires { T::ORDER; };

/// A representation of an element in a prime field.
template <FieldSpec Spec>
class FieldElement
{
    using uint_type = std::remove_const_t<decltype(Spec::ORDER)>;
    static constexpr bool is_bn_accel = requires { Spec::BN_ACCELERATED; };
    static constexpr ModArith<uint_type, is_bn_accel> Fp{Spec::ORDER};

#if defined(AIRBENDER) && defined(__riscv)
    alignas(32) uint_type value_;
#else
    uint_type value_;
#endif

    /// Wraps a value into the Element type assuming it is already in the internal ModArith form.
    [[gnu::always_inline]] static constexpr FieldElement wrap(const uint_type& v) noexcept
    {
        FieldElement element;
        element.value_ = v;
        return element;
    }

public:
    /// The alias to the finite field's order.
    static constexpr auto& ORDER = Spec::ORDER;

    FieldElement() = default;

    constexpr explicit FieldElement(uint_type v) : value_{Fp.to_mont(v)} {}

    constexpr uint_type value() const noexcept { return Fp.from_mont(value_); }

    static constexpr std::optional<FieldElement> from_bytes(
        std::span<const uint8_t, sizeof(uint_type)> b) noexcept
    {
        const auto x = intx::be::load<uint_type>(b);
        if (x >= ORDER) [[unlikely]]
            return std::nullopt;
        return FieldElement{x};
    }

    constexpr void to_bytes(std::span<uint8_t, sizeof(uint_type)> b) const noexcept
    {
        intx::be::store(b, value());
    }


    constexpr explicit operator bool() const noexcept { return static_cast<bool>(value_); }

    friend constexpr bool operator==(const FieldElement&, const FieldElement&) = default;

    friend constexpr bool operator==(const FieldElement& a, zero_t) noexcept { return !a.value_; }

    friend constexpr auto __attribute__((always_inline)) operator*(const FieldElement& a, const FieldElement& b) noexcept
    {
        return wrap(Fp.mul(a.value_, b.value_));
    }

    FieldElement& __attribute__((always_inline)) operator*=(const FieldElement& b) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        Fp.mul_assign(value_, b.value_);
#else
        value_ = Fp.mul(value_, b.value_);
#endif
        return *this;
    }

    friend constexpr auto __attribute__((always_inline)) operator+(const FieldElement& a, const FieldElement& b) noexcept
    {
        return wrap(Fp.add(a.value_, b.value_));
    }

    FieldElement& __attribute__((always_inline)) operator+=(const FieldElement& b) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        Fp.add_assign(value_, b.value_);
#else
        value_ = Fp.add(value_, b.value_);
#endif
        return *this;
    }

    FieldElement& __attribute__((always_inline)) operator-=(const FieldElement& b) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        Fp.sub_assign(value_, b.value_);
#else
        value_ = Fp.sub(value_, b.value_);
#endif
        return *this;
    }

    friend constexpr auto __attribute__((always_inline)) operator-(const FieldElement& a, const FieldElement& b) noexcept
    {
        return wrap(Fp.sub(a.value_, b.value_));
    }

    friend constexpr auto __attribute__((always_inline)) operator-(const FieldElement& a) noexcept
    {
        return wrap(Fp.sub(0, a.value_));
    }

    friend constexpr auto __attribute__((always_inline)) operator/(one_t, const FieldElement& a) noexcept
    {
        return wrap(Fp.inv(a.value_));
    }

    friend constexpr auto __attribute__((always_inline)) operator/(const FieldElement& a, const FieldElement& b) noexcept
    {
        return wrap(Fp.mul(a.value_, Fp.inv(b.value_)));
    }

    /// Named 1/x inversion method. Needed in the pairing templates.
    constexpr auto __attribute__((always_inline)) inv() const noexcept { return wrap(Fp.inv(value_)); }

    /// Repeated squaring: returns x^(2^n). Uses ModArith::square_n for CSR loop optimization.
    constexpr auto __attribute__((always_inline)) square_n(unsigned n) const noexcept { return wrap(Fp.square_n(value_, n)); }

    /// Named one element. Needed in the pairing templates.
    static constexpr auto one() noexcept { return FieldElement{1}; }
};

/// The affine (two coordinates) point on an Elliptic Curve over a prime field.
template <typename ValueT>
struct Point
{
    ValueT x = {};
    ValueT y = {};

    friend constexpr Point operator-(const Point& p) noexcept { return {p.x, -p.y}; }
};

/// The affine (two coordinates) point on an Elliptic Curve over a prime field.
template <typename Curve>
struct AffinePoint
{
    using FE = Curve::Fp;

    FE x;
    FE y;

    AffinePoint() = default;
    constexpr AffinePoint(const FE& x_, const FE& y_) noexcept : x{x_}, y{y_} {}

    /// Create the point from literal values.
    consteval AffinePoint(const Curve::uint_type& x_value, const Curve::uint_type& y_value) noexcept
      : x{x_value}, y{y_value}
    {}

    friend constexpr AffinePoint operator-(const AffinePoint& p) noexcept
    {
        return {p.x, -p.y};
    }

    friend constexpr bool operator==(const AffinePoint&, const AffinePoint&) = default;

    friend constexpr bool operator==(const AffinePoint& p, zero_t) noexcept
    {
        return p == AffinePoint{};
    }

    static constexpr std::optional<AffinePoint> from_bytes(
        std::span<const uint8_t, sizeof(FE) * 2> b) noexcept
    {
        const auto x = FE::from_bytes(b.template subspan<0, sizeof(FE)>());
        const auto y = FE::from_bytes(b.template subspan<sizeof(FE), sizeof(FE)>());
        if (!x.has_value() || !y.has_value()) [[unlikely]]
            return std::nullopt;
        return AffinePoint{*x, *y};
    }

    constexpr void to_bytes(std::span<uint8_t, sizeof(FE) * 2> b) const noexcept
    {
        x.to_bytes(b.template subspan<0, sizeof(FE)>());
        y.to_bytes(b.template subspan<sizeof(FE), sizeof(FE)>());
    }
};

/// Elliptic curve point in Jacobian coordinates (X, Y, Z)
/// representing the affine point (X/Z², Y/Z³).
/// TODO: Merge with JacPoint.
template <typename Curve>
struct ProjPoint
{
    using FE = Curve::Fp;
    FE x;
    FE y{1};  // TODO: Make sure this is compile-time constant.
    FE z;

    ProjPoint() = default;
    constexpr ProjPoint(const FE& x_, const FE& y_, const FE& z_) noexcept : x{x_}, y{y_}, z{z_} {}
    constexpr explicit ProjPoint(const AffinePoint<Curve>& p) noexcept : x{p.x}, y{p.y}, z{FE{1}} {}

    friend constexpr bool operator==(const ProjPoint& p, zero_t) noexcept { return p.z == 0; }

    friend constexpr bool operator==(const ProjPoint& p, const ProjPoint& q) noexcept
    {
        const auto& [x1, y1, z1] = p;
        const auto& [x2, y2, z2] = q;
        const auto z1z1 = z1 * z1;
        const auto z1z1z1 = z1z1 * z1;
        const auto z2z2 = z2 * z2;
        const auto z2z2z2 = z2z2 * z2;
        return x1 * z2z2 == x2 * z1z1 && y1 * z2z2z2 == y2 * z1z1z1;
    }

    friend constexpr ProjPoint operator-(const ProjPoint& p) noexcept { return {p.x, -p.y, p.z}; }
};

// Jacobian (three) coordinates point implementation.
template <typename ValueT>
struct JacPoint
{
    ValueT x = 1;
    ValueT y = 1;
    ValueT z = 0;

    // Compares two Jacobian coordinates points
    friend constexpr bool operator==(const JacPoint& a, const JacPoint& b) noexcept
    {
        const auto bz2 = b.z * b.z;
        const auto az2 = a.z * a.z;

        const auto bz3 = bz2 * b.z;
        const auto az3 = az2 * a.z;

        return a.x * bz2 == b.x * az2 && a.y * bz3 == b.y * az3;
    }

    friend constexpr JacPoint operator-(const JacPoint& p) noexcept { return {p.x, -p.y, p.z}; }

    // Creates Jacobian coordinates point from affine point
    static constexpr JacPoint from(const ecc::Point<ValueT>& ap) noexcept
    {
        return {ap.x, ap.y, ValueT::one()};
    }
};

template <typename IntT>
using InvFn = IntT (*)(const ModArith<IntT>&, const IntT& x) noexcept;

/// Converts a projected point to an affine point.
template <typename Curve>
__attribute__((flatten))
inline AffinePoint<Curve> to_affine(const ProjPoint<Curve>& p) noexcept
{
    // This works correctly for the point at infinity (z == 0) because then z_inv == 0.
    auto z_inv = 1 / p.z;
    auto zz_inv = z_inv; zz_inv *= z_inv;  // z_inv^2 (copy+mul_assign saves 1 MEMCOPY)
    z_inv *= zz_inv;            // z_inv now = zzz_inv = zz_inv * z_inv (in-place, saves 1 MEMCOPY)
    auto rx = p.x; rx *= zz_inv;   // x/z^2 (copy+mul_assign saves 1 MEMCOPY)
    auto ry = p.y; ry *= z_inv;    // y/z^3 (copy+mul_assign saves 1 MEMCOPY)
    return {rx, ry};
}

/// Elliptic curve point addition in affine coordinates.
///
/// Computes P ⊕ Q for two points in affine coordinates on the elliptic curve.
/// This procedure handles all inputs (e.g. doubling or points at infinity).
/// The produced result is also in affine coordinates. Therefore, this is useful only for one-off
/// additions, as otherwise multiple additions would be inefficient due to repeated inversions.
template <typename Curve>
AffinePoint<Curve> add_affine(const AffinePoint<Curve>& p, const AffinePoint<Curve>& q) noexcept
{
    if (p == 0)
        return q;
    if (q == 0)
        return p;

    const auto& [x1, y1] = p;
    const auto& [x2, y2] = q;

    // Use classic formula for point addition.
    // https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_operations

    auto dx = x2 - x1;
    auto dy = y2 - y1;
    if (dx == 0)
    {
        if (dy != 0)    // For opposite points
            return {};  // return the point at infinity.

        // For coincident points find the slope of the tangent line.
        const auto xx = x1 * x1;
        dy = xx + xx + xx;
        if constexpr (Curve::A != 0)
            dy += typename Curve::Fp{Curve::A};
        dx = y1 + y1;
    }
    const auto slope = dy / dx;

    const auto xr = slope * slope - x1 - x2;
    const auto yr = slope * (x1 - xr) - y1;
    return {xr, yr};
}

/// Elliptic curve point addition in Jacobian coordinates.
///
/// Computes P ⊕ Q for two points in Jacobian coordinates on the elliptic curve.
/// This procedure handles all inputs (e.g. doubling or points at infinity).
template <typename Curve>
__attribute__((flatten))
ProjPoint<Curve> add(const ProjPoint<Curve>& p, const ProjPoint<Curve>& q) noexcept
{
    if (p == 0)
        return q;
    if (q == 0)
        // TODO: Untested and untestable via precompile call (for secp256k1 and secp256r1).
        return p;

    // Use the "add-1998-cmo-2" formula for curve in Jacobian coordinates.
    // The cost is 12M + 4S + 6add + 1*2.
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#addition-add-1998-cmo-2
    // TODO: The newer formula "add-2007-bl" trades one multiplication for one squaring and
    //   additional additions. We don't have dedicated squaring operation yet, so it's not clear
    //   if it would be faster.

    const auto& [x1, y1, z1] = p;
    const auto& [x2, y2, z2] = q;

    auto z1z1 = z1; z1z1 *= z1;    // z1^2 (copy+mul_assign saves 1 MEMCOPY)
    auto z2z2 = z2; z2z2 *= z2;    // z2^2 (copy+mul_assign saves 1 MEMCOPY)
    auto u1 = x1; u1 *= z2z2;     // x1*z2^2 (copy+mul_assign saves 1 MEMCOPY)
    auto u2 = x2; u2 *= z1z1;     // x2*z1^2 (copy+mul_assign saves 1 MEMCOPY)
    z1z1 *= z1;                 // z1z1 now = z1^3 (saves 1 MEMCOPY)
    z2z2 *= z2;                 // z2z2 now = z2^3 (saves 1 MEMCOPY)
    z2z2 *= y1;                 // z2z2 now = s1 = y1*z2^3 (saves 1 MEMCOPY)
    z1z1 *= y2;                 // z1z1 now = s2 = y2*z1^3 (saves 1 MEMCOPY)
    u2 -= u1;                  // u2 now = h = u2 - u1 (in-place, saves 1 MEMCOPY)
    auto& h = u2;
    z1z1 -= z2z2;              // z1z1 now = r = s2 - s1 (in-place, saves 1 MEMCOPY)
    auto& r = z1z1;

    // Handle point doubling in case p == q, i.e. when u1 == u2 and s1 == s2.
    // TODO: Untested case of two points having the same y coordinate but different x.
    //       The following assertion (r == 0) => (h == 0) should fail in that case.
    assert(r != 0 || h == 0);
    if (h == 0 && r == 0) [[unlikely]]
        return dbl(p);

    auto hh = h; hh *= h;         // h^2 (copy+mul_assign saves 1 MEMCOPY)
    u1 *= hh;                  // u1 now = v = u1 * hh (in-place, saves 1 MEMCOPY)
    auto& v = u1;
    hh *= h;                    // hh now = hhh = h^3 (saves 1 MEMCOPY)
    auto x3 = r; x3 *= r;         // r^2 (copy+mul_assign saves 1 MEMCOPY)
    auto t3 = v;
    t3 += v;                   // t3 = 2*v    (in-place, saves 1 MEMCOPY)
    x3 -= hh;                 // t4 = t2 - hhh (in-place, saves 1 MEMCOPY)
    x3 -= t3;                 // x3 = t4 - t3  (in-place, saves 1 MEMCOPY)
    v -= x3;                   // v now = v - x3 (in-place, eliminates temporary t5)
    hh *= z2z2;                // hh now = s1*hhh = t6 (saves 1 MEMCOPY; z2z2 holds s1)
    r *= v;                    // r now = r*(v-x3) = t7 (saves 1 MEMCOPY)
    r -= hh;                   // r now = y3 = t7 - t6  (in-place, saves 1 MEMCOPY)
    h *= z2;                   // h now = z2*h = t8 (saves 1 MEMCOPY)
    h *= z1;                   // h now = z3 = z1*t8 (saves 1 MEMCOPY)

    return {x3, r, h};
}

/// Mixed addition of elliptic curve points.
///
/// Computes P ⊕ Q for a point P in Jacobian coordinates and a point Q in affine coordinates.
/// This procedure handles all inputs (e.g. doubling or points at infinity).
template <typename Curve>
__attribute__((flatten))
ProjPoint<Curve> add(const ProjPoint<Curve>& p, const AffinePoint<Curve>& q) noexcept
{
    if (q == 0)
        // TODO: Untested and untestable via precompile call (for secp256r1).
        return p;
    if (p == 0)
        return ProjPoint(q);

    // Use the "madd" formula for curve in Jacobian coordinates.
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#addition-madd
    // Modified to properly support adding the same point.

    const auto& [x1, y1, z1] = p;
    const auto& [x2, y2] = q;

    auto z1z1 = z1; z1z1 *= z1;    // z1^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
    auto u2 = x2; u2 *= z1z1;     // x2*z1^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
    z1z1 *= z1;                 // z1z1 now = z1^3 (saves 1 MEMCOPY vs z1z1z1 = z1 * z1z1)
    z1z1 *= y2;                 // z1z1 now = s2 = y2 * z1^3 (saves 1 MEMCOPY vs s2 = y2 * z1z1z1)
    u2 -= x1;                   // u2 now = h = u2 - x1 (in-place, saves 1 MEMCOPY)
    auto& h = u2;
    auto t1 = h;
    t1 += h;                    // t1 = 2*h  (in-place, saves 1 MEMCOPY)
    auto i = t1; i *= t1;         // (2h)^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
    z1z1 -= y1;                 // z1z1 now = t2 = s2 - y1 (in-place, saves 1 MEMCOPY)
    auto& t2 = z1z1;

    // Handle point doubling in case p == q.
    // p == q (in jacobian coordinates) if and only if x1 == x2 * z1z1 and y1 = y2 * z1z1z1
    if (h == 0 && t2 == 0) [[unlikely]]
        return dbl(p);

    auto r = t2;
    r += t2;                    // r = 2*t2  (in-place, saves 1 MEMCOPY)
    auto v = x1; v *= i;          // x1*i (copy+mul_assign saves 1 MEMCOPY vs operator*)
    i *= h;                     // i now = j = h * i (saves 1 MEMCOPY vs j = h * i)
    auto x3 = r; x3 *= r;         // r^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
    auto t4 = v;
    t4 += v;                    // t4 = 2*v  (in-place, saves 1 MEMCOPY)
    x3 -= i;                   // t5 = t3 - j  (in-place, saves 1 MEMCOPY)
    x3 -= t4;                  // x3 = t5 - t4 (in-place, saves 1 MEMCOPY)
    v -= x3;                    // v now = v - x3 (in-place, eliminates temporary t6)
    i *= y1;                    // i now = y1 * j = t7 (saves 1 MEMCOPY vs t7 = y1 * j)
    r *= v;                     // r now = y3 = r * (v-x3) (saves 1 MEMCOPY)
    i += i;                     // t8 = 2*t7 (in-place, saves 1 MEMCOPY)
    r -= i;                     // y3 = t9 - t8 (in-place, saves 1 MEMCOPY)
    h *= z1;                    // h now = z3 = z1 * h (saves 1 MEMCOPY vs z3 = z1 * h)
    h += h;                     // z3 = 2*t10 (in-place, saves 1 MEMCOPY)

    return {x3, r, h};
}

template <typename Curve>
__attribute__((flatten))
ProjPoint<Curve> dbl(const ProjPoint<Curve>& p) noexcept
{
    const auto& [x1, y1, z1] = p;

    if constexpr (Curve::A == 0)
    {
        // Optimized doubling for a=0 curve in Jacobian coordinates.
        // Computes S = 4*X*Y^2 directly (1M + 2A) instead of via the dbl-2009-l
        // squaring trick (1M + 2S + 1A), saving 2 modular subtractions per doubling.
        // Formula: S = 4*X*Y^2, M = 3*X^2, X' = M^2 - 2S, Y' = M(S-X') - 8Y^4, Z' = 2YZ.
        // Cost: 7M + 9A + 3S = 7M + 12(A+S) vs original 7M + 9A + 5S = 7M + 14(A+S).

        auto xx = x1; xx *= x1;       // X^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
        auto yy = y1; yy *= y1;       // Y^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
        auto yyyy = yy; yyyy *= yy;   // Y^4 (copy+mul_assign saves 1 MEMCOPY vs operator*)
        yy *= x1;              // yy now = s = X*Y^2 (saves 1 MEMCOPY vs s = x1 * yy)
        yy += yy;              // s = 2*X*Y^2        (in-place, saves 1 MEMCOPY)
        yy += yy;              // S = 4*X*Y^2        (in-place, saves 1 MEMCOPY)
        auto m = xx;            // copy X^2
        m += xx;                // 2*X^2             (in-place, saves 1 MEMCOPY vs m = xx + xx)
        m += xx;                // M = 3*X^2         (in-place)
        auto x3 = m; x3 *= m;         // M^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
        x3 -= yy;              // M^2 - S   (eliminates s2 copy: was `s2=yy; s2+=yy; x3-=s2`)
        x3 -= yy;              // X' = M^2 - 2*S    (in-place)
        yy -= x3;              // yy now = S - X'   (in-place, eliminates temporary t)
        yyyy += yyyy;           // 2*Y^4             (in-place, saves 1 MEMCOPY)
        yyyy += yyyy;           // 4*Y^4             (in-place, saves 1 MEMCOPY)
        yyyy += yyyy;           // 8*Y^4             (in-place, saves 1 MEMCOPY)
        m *= yy;                // m now = Y' = M*(S-X') (saves 1 MEMCOPY vs y3 = m * t)
        m -= yyyy;             // Y' = M*(S - X') - 8*Y^4  (in-place, saves 1 MEMCOPY)
        auto z3 = y1; z3 *= z1;       // Y*Z (copy+mul_assign saves 1 MEMCOPY vs operator*)
        z3 += z3;              // Z' = 2*Y*Z        (in-place, saves 1 MEMCOPY)
        return {x3, m, z3};
    }
    else if constexpr (Curve::A == Curve::FIELD_PRIME - 3)
    {
        // Use the "dbl-2001-b" doubling formula.
        // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2001-b

        const auto zz = z1 * z1;
        const auto yy = y1 * y1;
        auto xyy = x1 * yy;
        const auto t0 = x1 - zz;
        const auto t1 = x1 + zz;
        const auto t2 = t0 * t1;
        auto alpha = t2;
        alpha += t2;               // 2*t2     (in-place)
        alpha += t2;               // alpha = 3*t2 (in-place)
        auto x3 = alpha * alpha;   // t3
        xyy += xyy;               // xyy2     (in-place)
        xyy += xyy;               // xyy4     (in-place)
        auto xyy4_save = xyy;     // save for t9
        xyy += xyy;               // xyy8     (in-place)
        x3 -= xyy;               // x3 = t3 - xyy8 (in-place)
        const auto t5 = y1 + z1;
        auto z3 = t5 * t5;       // t6
        z3 -= yy;                // t7       (in-place)
        z3 -= zz;                // z3       (in-place)
        xyy4_save -= x3;         // t9 = xyy4 - x3 (in-place)
        auto yyyy = yy * yy;
        yyyy += yyyy;             // yyyy2    (in-place)
        yyyy += yyyy;             // yyyy4    (in-place)
        yyyy += yyyy;             // yyyy8    (in-place)
        alpha *= xyy4_save;          // alpha now = t12 (saves 1 MEMCOPY)
        alpha -= yyyy;               // y3       (in-place)
        return {x3, alpha, z3};
    }
    else
    {
        // TODO(c++23): Use fake always-false condition for older compilers.
        static_assert(Curve::A == 0, "unsupported Curve::A value");
    }
}

/// Tests if a specific bit is set in an integer type.
/// Bits are indexed from the least significant. Checking beyond the bit-width is undefined.
/// TODO: Move to intx.
template <typename IntT>
bool test_bit(const IntT& v, size_t bit_index) noexcept
{
    using word_type = IntT::word_type;
    static constexpr auto WORD_BITS = sizeof(word_type) * 8;
    const auto word = v[(bit_index / WORD_BITS)];
    const auto b = bit_index % WORD_BITS;
    return (word & (uint64_t{1} << b)) != 0;
}

template <typename Curve>
ProjPoint<Curve> mul(const AffinePoint<Curve>& p, typename Curve::uint_type c) noexcept
{
    // Reduce the scalar by the curve group order.
    // This allows using more efficient add algorithm in the loop because doubling cannot happen.
    while (true)
    {
        const auto [reduced_c, less_than] = subc(c, Curve::ORDER);
        if (less_than) [[likely]]
            break;
        // TODO: Untested and untestable via precompile call (for secp256r1).
        c = reduced_c;
    }

    ProjPoint<Curve> r;
    for (size_t i = bit_width(c); i != 0; --i)
    {
        r = ecc::dbl(r);
        if (bit_test(c, i - 1))
            r = ecc::add(r, p);
    }
    return r;
}

/// Computes multi-scalar multiplication of u×P ⊕ v×Q.
///
/// The implementation uses the "Straus-Shamir trick": https://eprint.iacr.org/2003/257.pdf#page=7.
template <typename Curve>
ProjPoint<Curve> msm(const typename Curve::uint_type& u, const AffinePoint<Curve>& p,
    const typename Curve::uint_type& v, const AffinePoint<Curve>& q)
{
    ProjPoint<Curve> r;

    const auto bit_width = intx::bit_width(u | v);
    if (bit_width == 0)
        return r;

    // Precompute affine P + Q. Works correctly if P == Q.
    const auto h = add_affine(p, q);

    // Create lookup table for points. The index 0 is unused.
    // TODO: Put 0 at index 0 and use it in the loop to avoid the branch.
    const AffinePoint<Curve>* const points[]{nullptr, &p, &q, &h};

    for (auto i = bit_width; i != 0; --i)
    {
        r = dbl(r);

        const auto u_bit = bit_test(u, i - 1);
        const auto v_bit = bit_test(v, i - 1);
        const auto idx = 2 * size_t{v_bit} + size_t{u_bit};
        if (idx == 0)
            continue;
        r = add(r, *points[idx]);
    }

    return r;
}

template <typename UIntT>
struct SignedScalar
{
    bool sign = false;  // The sign of the scalar: false = positive, true = negative.
    UIntT value;
};


/// Verifies k ≡ k₁ + k₂·λ (mod N) and checks that k₁ and k₂ are "short" scalars.
template <typename Curve>
[[maybe_unused, nodiscard]] bool verify_scalar_decomposition(const typename Curve::uint_type& k,
    const SignedScalar<typename Curve::uint_type>& k1,
    const SignedScalar<typename Curve::uint_type>& k2) noexcept
{
    // Verify k ≡ k₁ + k₂·λ (mod N).
    {
        static constexpr ModArith N{Curve::ORDER};
        auto r_k1 = N.to_mont(k1.value);
        if (k1.sign)
            r_k1 = N.sub(0, r_k1);
        auto r_k2 = N.to_mont(k2.value);
        if (k2.sign)
            r_k2 = N.sub(0, r_k2);

        const auto r_k = N.to_mont(k);

        const auto right = N.add(r_k1, N.mul(r_k2, N.to_mont(Curve::LAMBDA)));
        if (r_k != right)
            return false;
    }

    // Verify for u = (k₁, k₂) that ‖u‖ <= max(‖v₁‖, ‖v₂‖).
    {
        static constexpr auto V1_NORM_SQUARED =
            Curve::X1 * Curve::X1 + Curve::MINUS_Y1 * Curve::MINUS_Y1;
        static constexpr auto V2_NORM_SQUARED = Curve::X2 * Curve::X2 + Curve::Y2 * Curve::Y2;
        static constexpr auto MAX_NORM_SQUARED = std::max(V1_NORM_SQUARED, V2_NORM_SQUARED);
        const auto u_norm_squared = k1.value * k1.value + k2.value * k2.value;
        return u_norm_squared <= MAX_NORM_SQUARED;
    }
}

/// Decomposes a scalar k into "short" scalars k₁ and k₂ such that k₁ + k₂·λ ≡ k (mod N).
///
/// This decomposition allows more efficient scalar multiplication by using the multi-scalar
/// multiplication (MSM) and the GLV endomorphism.
/// The endomorphism ϕ: E → E defined as (x,y) → (βx,y) with eigenvalue λ allows computing
/// [λ](x,y) = (βx,y) with only one multiplication in 𝔽ₚ instead of a full scalar multiplication.
///
/// Moreover, to compute the short scalars k₁ and k₂, we need linearly independent short vectors
/// (v₁=(x₁,y₁), v₂=(x₂,y₂)) such that f(v₁) = f(v₂) = 0,
/// where f: ℤ×ℤ → ℤₙ is defined as (x,y) → (x + y·λ), where λ² + λ ≡ -1 mod N.
///
/// See https://www.iacr.org/archive/crypto2001/21390189.pdf for details.
///
/// The Curve type must provide the endomorphism parameters: LAMBDA, BETA, X1, MINUS_Y1, X2, Y2.
template <typename Curve>
std::array<SignedScalar<typename Curve::uint_type>, 2> decompose(
    const typename Curve::uint_type& k) noexcept
{
    using UIntT = Curve::uint_type;

    // Validate the provided setup parameters.
    // λ² + λ ≡ -1 mod n
    static_assert((umul(Curve::LAMBDA, Curve::LAMBDA) + Curve::LAMBDA + 1) % Curve::ORDER == 0);
    // f: (x, y) → (x + λy) mod N
    // f(v₁) = 0
    static_assert(
        (Curve::X1 + umul(Curve::ORDER - Curve::MINUS_Y1, Curve::LAMBDA)) % Curve::ORDER == 0);
    // f(v₂) = 0
    static_assert((Curve::X2 + umul(Curve::Y2, Curve::LAMBDA)) % Curve::ORDER == 0);

    // DET is the (v₁, v₂) matrix determinant.
    static constexpr auto WIDE_DET =
        umul(Curve::X1, Curve::Y2) + umul(Curve::X2, Curve::MINUS_Y1);
    static_assert(WIDE_DET <= std::numeric_limits<UIntT>::max());
    static constexpr auto DET = static_cast<UIntT>(WIDE_DET);
    static constexpr auto HALF_DET = DET / 2;

#if defined(AIRBENDER) && defined(__riscv)
    // Barrett reduction constants for replacing expensive udivrem(uint512, uint256).
    // M = floor(2^512 / DET) = 2^256 + M_LO, where M_LO = floor((2^256-DET)*2^256 / DET).
    // Only usable when M_LO fits in 256 bits (i.e. DET has full 256-bit width).
    static constexpr auto BARRETT_GAP = ~DET + UIntT{1};  // 2^256 - DET
    static constexpr auto BARRETT_M_LO_WIDE =
        (intx::uint<512>{BARRETT_GAP} << 256) / intx::uint<512>{DET};
    static constexpr bool BARRETT_FITS =
        BARRETT_M_LO_WIDE <= std::numeric_limits<UIntT>::max();
    // Only define M_LO when it fits; otherwise Barrett path is disabled.
    static constexpr auto BARRETT_M_LO = BARRETT_FITS
        ? static_cast<UIntT>(BARRETT_M_LO_WIDE) : UIntT{};
#endif

    static constexpr auto round_div = [](const auto& a) noexcept {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (BARRETT_FITS)
        {
            if (!std::is_constant_evaluated())
            {
                // Barrett reduction: replace expensive udivrem(uint512, uint256) with
                // one CSR MUL_HIGH (for approximate quotient) + one umul (for correction).
                //
                // q_hat = a_hi + MUL_HIGH(a_hi, M_LO) is an approximate quotient
                // satisfying q_hat <= q_true <= q_hat + 1.

                // Extract upper and lower 256-bit halves of the 512-bit dividend.
                const auto a_hi = static_cast<UIntT>(a >> 256);
                const auto a_lo = static_cast<UIntT>(a);

                // Step 1: q_hat = a_hi + MUL_HIGH(a_hi, M_LO)
                alignas(32) UIntT buf_hi = a_hi;
                alignas(32) UIntT m_lo_buf = BARRETT_M_LO;
                {
                    register uintptr_t r10 asm("x10") =
                        reinterpret_cast<uintptr_t>(&buf_hi);
                    register uintptr_t r11 asm("x11") =
                        reinterpret_cast<uintptr_t>(&m_lo_buf);
                    register uint32_t r12 asm("x12") = 0x10;  // MUL_HIGH
                    asm volatile("csrrw x0, 0x7CA, x0"
                        : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
                }
                auto q_hat = a_hi + buf_hi;

                // Step 2: Compute diff = a - q_hat * DET (at most 2*DET - 1).
                // umul uses CSR MUL_LOW+MUL_HIGH on AIRBENDER.
                const auto product = umul(q_hat, DET);
                // 512-bit subtract (scalar on rv32im).
                const auto diff = a - product;
                auto diff_lo = static_cast<UIntT>(diff);

                // Step 3: Correction. If diff >= DET, q_hat was 1 too low.
                const bool needs_correction =
                    static_cast<UIntT>(diff >> 256) != 0 || diff_lo >= DET;
                if (needs_correction)
                {
                    diff_lo -= DET;
                    q_hat += UIntT{1};
                }

                return q_hat + UIntT{diff_lo > HALF_DET};
            }
        }
#endif

        const auto [wide_q, r] = udivrem(a, DET);
        // Division reduces the quotient enough to fit into a single uint.
        // This can be shown at compile-time by inspecting the DET and Y2/-Y1 values.
        assert(wide_q < std::numeric_limits<UIntT>::max());
        const auto q = static_cast<UIntT>(wide_q);
        return q + (r > HALF_DET);  // Round to nearest.
    };

    // Solve a system of two equations using Cramer method.
    // ⎡X1 X2⎤ * ⎡b1⎤ = ⎡k⎤
    // ⎣Y1 Y2⎦   ⎣b2⎦   ⎣0⎦
    // and then approximate to the nearest integers:
    // b1 = ⌊ Y2·k ÷ DET⌉
    // b2 = ⌊-Y1·k ÷ DET⌉
    const auto b1 = round_div(umul(k, Curve::Y2));
    const auto b2 = round_div(umul(k, Curve::MINUS_Y1));

    // k1 = k - (x1*b1 + x2*b2)
    const auto x1b1_x2b2 = umul(b1, Curve::X1) + umul(b2, Curve::X2);
    const auto [wide_k1, k1_is_neg] = subc(decltype(x1b1_x2b2){k}, x1b1_x2b2);
    const auto k1_abs = k1_is_neg ? -static_cast<UIntT>(wide_k1) : static_cast<UIntT>(wide_k1);

    // k2 = 0 - (y1*b1 + y2*b2)
    const auto [wide_k2, k2_is_neg] = subc(umul(b1, Curve::MINUS_Y1), umul(b2, Curve::Y2));
    const auto k2_abs = k2_is_neg ? -static_cast<UIntT>(wide_k2) : static_cast<UIntT>(wide_k2);

    const SignedScalar k1{k1_is_neg, k1_abs};
    const SignedScalar k2{k2_is_neg, k2_abs};
    assert(verify_scalar_decomposition<Curve>(k, k1, k2));
    return {k1, k2};
}

}  // namespace evmmax::ecc
