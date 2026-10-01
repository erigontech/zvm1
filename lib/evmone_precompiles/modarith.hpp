// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2023 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <intx/intx.hpp>
#include <cassert>

#ifdef SP1
#include <sp1_syscalls.hpp>
#endif

namespace evmone::crypto
{
/// Computes the modular inverse of a modulo 2³², satisfying a⋅inv = 1 mod 2³².
/// Only odd arguments are invertible modulo a power of two.
constexpr uint32_t modinv_pow2(uint32_t a) noexcept
{
    assert(a % 2 == 1);  // The argument must be odd, otherwise the inverse does not exist.

    // Start with inversion mod 2⁴, which is a³ mod 2⁴ (for odd a).
    // All 8 cases can be verified manually, but the formal explanation can be found in:
    // https://en.wikipedia.org/wiki/Multiplicative_group_of_integers_modulo_n#Powers_of_2.
    // This is better tradeoff than a mod 2² plus one Newton-Raphson iteration.
    // We also avoid explicit mod 2⁴ because the top garbage bits are fine.
    auto inv = a * a * a;

    // Use the Newton–Raphson numeric method, see e.g.
    // https://gmplib.org/~tege/divcnst-pldi94.pdf#page=9, formula (9.2)
    // Each iteration doubles the number of correct bits, starting from 4:
    // 8, 16, 32, ..., so for 32-bit value we need 3 iterations.
    // TODO(C++23): static
    constexpr auto ITERATIONS = std::countr_zero(sizeof(a) * 8 / 4);
    for (auto i = 0; i < ITERATIONS; ++i)
        inv *= 2 - a * inv;  // Overflows are fine because they wrap around modulo 2³².

    assert(inv * a == 1);  // Verify the result.
    return inv;
}

/// Computes the modular inverse of a modulo 2⁶⁴, satisfying a⋅inv = 1 mod 2⁶⁴.
/// Only odd arguments are invertible modulo a power of two.
constexpr uint64_t modinv_pow2(uint64_t a) noexcept
{
    assert(a % 2 == 1);  // The argument must be odd, otherwise the inverse does not exist.
    uint64_t inv = modinv_pow2(static_cast<uint32_t>(a));  // Start with inversion mod 2³².
    inv *= 2 - a * inv;    // One Newton-Raphson iteration: 64 bits correct.
    assert(inv * a == 1);  // Verify the result.
    return inv;
}

/// Compute the modulus inverse for Montgomery multiplication, i.e., N': mod⋅N' = 2⁶⁴-1.
template <typename UintT>
constexpr uint64_t compute_mont_mod_inv(const UintT& mod) noexcept
{
    // Compute the inversion mod[0]⁻¹ mod 2⁶⁴, then the final result is N' = -mod[0]⁻¹
    // because this gives mod⋅N' = -1 mod 2⁶⁴ = 2⁶⁴-1.
    return -modinv_pow2(mod[0]);
}

constexpr std::pair<uint64_t, uint64_t> addmul(
    uint64_t t, uint64_t a, uint64_t b, uint64_t c) noexcept
{
    const auto p = intx::umul(a, b) + t + c;
    return {p[1], p[0]};
}

/// Performs a modular addition. Requires x < mod and y < mod.
template <typename UintT>
constexpr UintT modadd(const UintT& x, const UintT& y, const UintT& mod) noexcept
{
    assert(x < mod);
    assert(y < mod);
    const auto s = addc(x, y);  // TODO: cannot overflow if modulus is sparse (e.g. 255 bits).
    const auto d = subc(s.value, mod);
    return (!s.carry && d.carry) ? s.value : d.value;
}

/// Performs a modular subtraction. Requires x < mod and y < mod.
template <typename UintT>
constexpr UintT modsub(const UintT& x, const UintT& y, const UintT& mod) noexcept
{
    assert(x < mod);
    assert(y < mod);
    const auto d = subc(x, y);
    const auto s = d.value + mod;
    return (d.carry) ? s : d.value;
}

/// Computes scale⋅x⁻¹ % mod, i.e. the modular inverse of x scaled by the given factor.
/// Returns 0 for non-invertible x (including x == 0).
/// Requires an odd mod not less than 3, and both x and scale less than mod.
///
/// The scale is the initial value of the Bézout coefficient u and the algorithm is linear in it,
/// so any factor can be folded into the result for free.
template <typename UintT>
constexpr UintT modinv_scaled(const UintT& x, const UintT& scale, const UintT& mod) noexcept
{
    assert((mod & 1) == 1);
    assert(mod >= 3);
    assert(x < mod);
    assert(scale < mod);  // Otherwise the first modsub() below gets a non-reduced operand.

    // Precompute inverse of 2 modulo mod: inv2 * 2 % mod == 1.
    // The 1/2 is inexact division that can be fixed by adding "0" to the numerator
    // and making it even: (mod + 1) / 2. To avoid potential overflow of (1 + mod)
    // we rewrite it further to (mod - 1 + 2) / 2 = (mod - 1) / 2 + 1 = ⌊mod / 2⌋ + 1.
    const auto inv2 = (mod >> 1) + 1;

    // Use extended binary Euclidean algorithm. This evolves variables a and b until a is 0.
    // Then GCD(x, mod) is in b. If GCD(x, mod) == 1 then the inversion exists and is in v.
    // This follows the classic algorithm (Algorithm 1) presented in
    // "Optimized Binary GCD for Modular Inversion".
    // https://eprint.iacr.org/2020/972.pdf#algorithm.1
    // TODO: The same paper has additional optimizations that could be applied.
    UintT a = x;
    UintT b = mod;
    UintT u = scale;
    UintT v = 0;

    while (a != 0)
    {
        if ((a & 1) != 0)
        {
            // if a is odd, update it to a - b.
            if (const auto [d, less] = subc(a, b); less)
            {
                // swap a and b in case a < b.
                b = a;
                a = -d;

                using namespace std;
                swap(u, v);
            }
            else
            {
                a = d;
            }
            u = modsub(u, v, mod);
        }

        // Compute a / 2 % mod, a is even so division is exact and can be computed as ⌊a / 2⌋.
        a >>= 1;

        // Compute u / 2 % mod. If u is even, this can be computed as ⌊u / 2⌋.
        // Otherwise, (u - 1 + 1) / 2 = ⌊u / 2⌋ + (1 / 2 % mod).
        const auto u_odd = (u & 1) != 0;
        u >>= 1;
        if (u_odd)
            u += inv2;  // if u is odd, add back ½ % mod.
    }

    if (b != 1) [[unlikely]]
        v = 0;  // not invertible
    return v;
}

#if defined(__SIZEOF_INT128__)
namespace detail
{
/// Modular inversion for 256-bit odd moduli using the safegcd (divsteps) algorithm.
/// Ported from libsecp256k1's modinv64_var:
/// Copyright (c) 2020 Peter Dettman, MIT license,
/// https://github.com/bitcoin-core/secp256k1/blob/master/src/modinv64_impl.h
/// Inputs here are public (signatures, precompile operands), so variable time is fine; it
/// also measures faster than the branchless 59-step variant on SP1.

/// Signed 62-bit limb representation: value = sum(v[i]*2^(62*i)).
struct Signed62
{
    int64_t v[5];
};

/// Transition matrix of 62 divsteps: [u v; q r], determinant 2^62.
struct Trans2x2
{
    int64_t u, v, q, r;
};

inline constexpr int64_t M62 = static_cast<int64_t>(UINT64_MAX >> 2);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"  // __int128 is a GNU extension.
using int128 = __int128;
#pragma GCC diagnostic pop

constexpr Signed62 to_signed62(const intx::uint256& a) noexcept
{
    return {{static_cast<int64_t>(a[0] & static_cast<uint64_t>(M62)),
        static_cast<int64_t>(((a[0] >> 62) | (a[1] << 2)) & static_cast<uint64_t>(M62)),
        static_cast<int64_t>(((a[1] >> 60) | (a[2] << 4)) & static_cast<uint64_t>(M62)),
        static_cast<int64_t>(((a[2] >> 58) | (a[3] << 6)) & static_cast<uint64_t>(M62)),
        static_cast<int64_t>(a[3] >> 56)}};
}

/// Valid only for the normalized form (all limbs in [0, 2^62)).
constexpr intx::uint256 from_signed62(const Signed62& a) noexcept
{
    return intx::uint256{static_cast<uint64_t>(a.v[0])} |
           (intx::uint256{static_cast<uint64_t>(a.v[1])} << 62) |
           (intx::uint256{static_cast<uint64_t>(a.v[2])} << 124) |
           (intx::uint256{static_cast<uint64_t>(a.v[3])} << 186) |
           (intx::uint256{static_cast<uint64_t>(a.v[4])} << 248);
}

/// Branchless count-trailing-zeros via De Bruijn multiplication: rv64im has no ctz
/// instruction and std::countr_zero lowers to a libgcc call. x must be non-zero.
constexpr int ctz64(uint64_t x) noexcept
{
    constexpr uint64_t deb = 0x022FDD63CC95386Dull;
    constexpr uint8_t tab[64] = {0, 1, 2, 53, 3, 7, 54, 27, 4, 38, 41, 8, 34, 55, 48, 28, 62, 5,
        39, 46, 44, 42, 22, 9, 24, 35, 59, 56, 49, 18, 29, 11, 63, 52, 6, 26, 37, 40, 33, 47, 61,
        45, 43, 21, 23, 58, 17, 10, 51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13,
        12};
    return tab[((x & (0 - x)) * deb) >> 58];
}

/// Performs 62 divsteps on the low words f0, g0, accumulating the transition matrix in t.
/// Variable time: runs of zero bits of g are consumed at once, and a small multiple of f
/// cancelling up to 6 (or 4) low bits of g is derived per iteration. Retires fewer
/// instructions per processed divstep than the branchless 59-step variant, which is what
/// counts under SP1's flat per-instruction cost.
constexpr int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, Trans2x2& t) noexcept
{
    uint64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0, m = 0;
    uint32_t w = 0;
    int i = 62, limit = 0, zeros = 0;

    for (;;)
    {
        // Use a sentinel bit to count zeros only up to i.
        zeros = ctz64(g | (UINT64_MAX << i));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros;
        i -= zeros;
        if (i == 0)
            break;
        if (eta < 0)
        {
            uint64_t tmp = f;
            eta = -eta;
            f = g;
            g = 0 - tmp;
            tmp = u;
            u = q;
            q = 0 - tmp;
            tmp = v;
            v = r;
            r = 0 - tmp;
            // Cancel up to min(limit, 6) bottom bits of g by adding a multiple of f.
            limit = (static_cast<int>(eta) + 1) > i ? i : (static_cast<int>(eta) + 1);
            m = (UINT64_MAX >> (64 - limit)) & 63U;
            w = static_cast<uint32_t>((f * g * (f * f - 2)) & m);
        }
        else
        {
            // Cancel up to min(limit, 4) bottom bits of g.
            limit = (static_cast<int>(eta) + 1) > i ? i : (static_cast<int>(eta) + 1);
            m = (UINT64_MAX >> (64 - limit)) & 15U;
            w = static_cast<uint32_t>(f + (((f + 1) & 4) << 1));
            w = static_cast<uint32_t>((-uint64_t{w} * g) & m);
        }
        g += f * w;
        q += u * w;
        r += v * w;
    }
    t.u = static_cast<int64_t>(u);
    t.v = static_cast<int64_t>(v);
    t.q = static_cast<int64_t>(q);
    t.r = static_cast<int64_t>(r);
    return eta;
}

/// d, e := (t * [d, e] + modulus * [md, me]) / 2^62, with md, me chosen so the division is
/// exact and the results stay in range (-2*modulus, modulus).
constexpr void update_de_62(Signed62& d, Signed62& e, const Trans2x2& t, const Signed62& modulus,
    uint64_t modulus_inv62) noexcept
{
    const int64_t d0 = d.v[0], d1 = d.v[1], d2 = d.v[2], d3 = d.v[3], d4 = d.v[4];
    const int64_t e0 = e.v[0], e1 = e.v[1], e2 = e.v[2], e3 = e.v[3], e4 = e.v[4];
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    // [md, me] start as zero; plus [u, q] if d is negative; plus [v, r] if e is negative.
    const int64_t sd = d4 >> 63;
    const int64_t se = e4 >> 63;
    int64_t md = (u & sd) + (v & se);
    int64_t me = (q & sd) + (r & se);
    int128 cd = int128{u} * d0 + int128{v} * e0;
    int128 ce = int128{q} * d0 + int128{r} * e0;
    // Correct md, me so that t*[d,e]+modulus*[md,me] has 62 zero bottom bits.
    md -= static_cast<int64_t>((modulus_inv62 * static_cast<uint64_t>(cd) +
                                   static_cast<uint64_t>(md)) &
                               static_cast<uint64_t>(M62));
    me -= static_cast<int64_t>((modulus_inv62 * static_cast<uint64_t>(ce) +
                                   static_cast<uint64_t>(me)) &
                               static_cast<uint64_t>(M62));
    cd += int128{modulus.v[0]} * md;
    ce += int128{modulus.v[0]} * me;
    cd >>= 62;
    ce >>= 62;
    cd += int128{u} * d1 + int128{v} * e1;
    ce += int128{q} * d1 + int128{r} * e1;
    if (modulus.v[1] != 0)
    {
        cd += int128{modulus.v[1]} * md;
        ce += int128{modulus.v[1]} * me;
    }
    d.v[0] = static_cast<int64_t>(cd) & M62;
    cd >>= 62;
    e.v[0] = static_cast<int64_t>(ce) & M62;
    ce >>= 62;
    cd += int128{u} * d2 + int128{v} * e2;
    ce += int128{q} * d2 + int128{r} * e2;
    if (modulus.v[2] != 0)
    {
        cd += int128{modulus.v[2]} * md;
        ce += int128{modulus.v[2]} * me;
    }
    d.v[1] = static_cast<int64_t>(cd) & M62;
    cd >>= 62;
    e.v[1] = static_cast<int64_t>(ce) & M62;
    ce >>= 62;
    cd += int128{u} * d3 + int128{v} * e3;
    ce += int128{q} * d3 + int128{r} * e3;
    if (modulus.v[3] != 0)
    {
        cd += int128{modulus.v[3]} * md;
        ce += int128{modulus.v[3]} * me;
    }
    d.v[2] = static_cast<int64_t>(cd) & M62;
    cd >>= 62;
    e.v[2] = static_cast<int64_t>(ce) & M62;
    ce >>= 62;
    cd += int128{u} * d4 + int128{v} * e4;
    ce += int128{q} * d4 + int128{r} * e4;
    cd += int128{modulus.v[4]} * md;
    ce += int128{modulus.v[4]} * me;
    d.v[3] = static_cast<int64_t>(cd) & M62;
    cd >>= 62;
    e.v[3] = static_cast<int64_t>(ce) & M62;
    ce >>= 62;
    d.v[4] = static_cast<int64_t>(cd);
    e.v[4] = static_cast<int64_t>(ce);
}

/// f, g := t * [f, g] / 2^62 over the len active limbs (the division is exact).
constexpr void update_fg_62_var(int len, Signed62& f, Signed62& g, const Trans2x2& t) noexcept
{
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    int64_t fi = f.v[0];
    int64_t gi = g.v[0];
    int128 cf = int128{u} * fi + int128{v} * gi;
    int128 cg = int128{q} * fi + int128{r} * gi;
    cf >>= 62;
    cg >>= 62;
    for (int i = 1; i < len; ++i)
    {
        fi = f.v[i];
        gi = g.v[i];
        cf += int128{u} * fi + int128{v} * gi;
        cg += int128{q} * fi + int128{r} * gi;
        f.v[i - 1] = static_cast<int64_t>(cf) & M62;
        cf >>= 62;
        g.v[i - 1] = static_cast<int64_t>(cg) & M62;
        cg >>= 62;
    }
    f.v[len - 1] = static_cast<int64_t>(cf);
    g.v[len - 1] = static_cast<int64_t>(cg);
}

/// Reduces r from range (-2*modulus, modulus) to [0, modulus), negating first if sign < 0.
constexpr void normalize_62(Signed62& r, int64_t sign, const Signed62& modulus) noexcept
{
    int64_t r0 = r.v[0], r1 = r.v[1], r2 = r.v[2], r3 = r.v[3], r4 = r.v[4];

    int64_t cond_add = r4 >> 63;
    r0 += modulus.v[0] & cond_add;
    r1 += modulus.v[1] & cond_add;
    r2 += modulus.v[2] & cond_add;
    r3 += modulus.v[3] & cond_add;
    r4 += modulus.v[4] & cond_add;
    const int64_t cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r1 += r0 >> 62;
    r0 &= M62;
    r2 += r1 >> 62;
    r1 &= M62;
    r3 += r2 >> 62;
    r2 &= M62;
    r4 += r3 >> 62;
    r3 &= M62;

    cond_add = r4 >> 63;
    r0 += modulus.v[0] & cond_add;
    r1 += modulus.v[1] & cond_add;
    r2 += modulus.v[2] & cond_add;
    r3 += modulus.v[3] & cond_add;
    r4 += modulus.v[4] & cond_add;
    r1 += r0 >> 62;
    r0 &= M62;
    r2 += r1 >> 62;
    r1 &= M62;
    r3 += r2 >> 62;
    r2 &= M62;
    r4 += r3 >> 62;
    r3 &= M62;

    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
}

/// Modular inverse of x mod an odd modulus >= 3, 0 <= x < mod.
/// Returns 0 when the inverse does not exist (gcd(x, mod) != 1, including x == 0).
///
/// @param seed  The initial Bezout coefficient, normally 1. The algorithm keeps the invariant
///              d⋅x ≡ seed⋅f and e⋅x ≡ seed⋅g (mod), so it terminates with d = ±seed⋅x⁻¹:
///              scaling the seed scales the result. Passing R² therefore inverts a Montgomery
///              value in place -- R²⋅(xR)⁻¹ = x⁻¹R -- with no conversion on either side,
///              the same trick inv_binary_gcd() uses. Must be < mod to keep e in bounds.
constexpr intx::uint256 modinv256(
    const intx::uint256& x, const intx::uint256& mod, const intx::uint256& seed) noexcept
{
    const Signed62 modulus = to_signed62(mod);
    const uint64_t modulus_inv62 = modinv_pow2(mod[0]) & static_cast<uint64_t>(M62);
    Signed62 d{{0, 0, 0, 0, 0}};
    Signed62 e = to_signed62(seed);
    Signed62 f = modulus;
    Signed62 g = to_signed62(x);
    int len = 5;
    int64_t eta = -1;  // eta = -delta; delta is initially 1.

    // 62 divsteps per iteration until g = 0; termination is bounded by ~741 divsteps
    // for 256-bit inputs (Bernstein-Yang 2019).
    while (true)
    {
        Trans2x2 t{};
        eta = divsteps_62_var(
            eta, static_cast<uint64_t>(f.v[0]), static_cast<uint64_t>(g.v[0]), t);
        update_de_62(d, e, t, modulus, modulus_inv62);
        update_fg_62_var(len, f, g, t);
        if (g.v[0] == 0)
        {
            int64_t cond = 0;
            for (int j = 1; j < len; ++j)
                cond |= g.v[j];
            if (cond == 0)
                break;
        }
        // Shrink the active length while the top limbs of f and g are redundant sign words.
        const int64_t fn = f.v[len - 1];
        const int64_t gn = g.v[len - 1];
        int64_t cond = (int64_t{len} - 2) >> 63;
        cond |= fn ^ (fn >> 63);
        cond |= gn ^ (gn >> 63);
        if (cond == 0)
        {
            f.v[len - 2] |= static_cast<int64_t>(static_cast<uint64_t>(fn) << 62);
            g.v[len - 2] |= static_cast<int64_t>(static_cast<uint64_t>(gn) << 62);
            --len;
        }
    }

    // g == 0 now, so f == +/- gcd(x, mod); d == +/- inverse when the gcd is 1.
    // x == 0 leaves f == mod, caught below. Normalizing f maps +1 -> 1 and -1 -> mod-1.
    Signed62 fr{{0, 0, 0, 0, 0}};
    for (int j = 0; j < len; ++j)
        fr.v[j] = f.v[j];
    normalize_62(fr, 0, modulus);
    const auto f_norm = from_signed62(fr);
    if (f_norm == 1)
        normalize_62(d, 0, modulus);
    else if (f_norm == mod - 1)
        normalize_62(d, -1, modulus);
    else
        return 0;
    return from_signed62(d);
}
}  // namespace detail
#endif  // defined(__SIZEOF_INT128__)


/// The modular arithmetic operations using the Montgomery form of the values.
template <typename UintT, bool BN = false>
class ModArith
{
    const UintT mod_;  ///< The modulus.

    const UintT r_squared_;  ///< R² % mod.

    /// The modulus inversion, i.e. the number N' such that mod⋅N' = 2⁶⁴-1.
    const uint64_t mod_inv_;

    /// Compute R² % mod.
    static constexpr UintT compute_r_squared(const UintT& mod) noexcept
    {
        // R is 2^num_bits, R² is 2^(2*num_bits) and needs 2*num_bits+1 bits to represent,
        // rounded to 2*num_bits+64 for intx requirements.
        constexpr auto RR = intx::uint<UintT::num_bits * 2 + 64>{1} << (UintT::num_bits * 2);
        return intx::udivrem(RR, mod).rem;
    }

public:
    constexpr explicit ModArith(const UintT& mod) noexcept
      : mod_{mod},
#if defined SP1 || defined SP1TURBO
        r_squared_{BN ? 1 : compute_r_squared(mod)},
        mod_inv_{BN ? 0 : compute_mont_mod_inv(mod)}
#else
        r_squared_{compute_r_squared(mod)},
        mod_inv_{compute_mont_mod_inv(mod)}
#endif
    {}

    /// Returns the modulus.
    constexpr const UintT& mod() const noexcept { return mod_; }

    /// Converts a value to Montgomery form.
    ///
    /// This is done by using Montgomery multiplication mul(x, R²)
    /// what gives aR²R⁻¹ % mod = aR % mod.
    constexpr UintT to_mont(const UintT& x) const noexcept
    {
#if defined SP1 || defined SP1TURBO
        if constexpr (BN)
            return x;
        else
#endif
            return mul(x, r_squared_);
    }

    /// Converts a value in Montgomery form back to normal value.
    ///
    /// Given the x is the Montgomery form x = aR, the conversion is done by using
    /// Montgomery multiplication mul(x, 1) what gives aRR⁻¹ % mod = a % mod.
    constexpr UintT from_mont(const UintT& x) const noexcept
    {
#if defined SP1 || defined SP1TURBO
        if constexpr (BN)
            return x;
        else
#endif
            return mul(x, 1);
    }

    /// Performs a Montgomery modular multiplication.
    ///
    /// Inputs must be in Montgomery form: x = aR, y = bR.
    /// This computes Montgomery multiplication xyR⁻¹ % mod what gives aRbRR⁻¹ % mod = abR % mod.
    /// The result (abR) is in Montgomery form.
    constexpr UintT mul(const UintT& x, const UintT& y) const noexcept
    {
#if defined(SP1) || defined(SP1TURBO)
        if constexpr (BN)
        {
            UintT res = x;
            syscall_bn254_fp_mulmod(
                reinterpret_cast<size_t*>(&res), reinterpret_cast<const size_t*>(&y));
            return res;
        }
#endif


        // Coarsely Integrated Operand Scanning (CIOS) Method
        // Based on 2.3.2 from
        // High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
        // https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf

        constexpr auto S = UintT::num_words;  // TODO(C++23): Make it static

        intx::uint<UintT::num_bits + 64> t;
        for (size_t i = 0; i != S; ++i)
        {
            uint64_t c = 0;
#pragma GCC unroll 8
            for (size_t j = 0; j != S; ++j)
                std::tie(c, t[j]) = addmul(t[j], x[j], y[i], c);
            auto tmp = intx::addc(t[S], c);
            t[S] = tmp.value;
            const auto d = tmp.carry;  // TODO: Carry is 0 for sparse modulus.

            const auto m = t[0] * mod_inv_;
            std::tie(c, std::ignore) = addmul(t[0], m, mod_[0], 0);
#pragma GCC unroll 8
            for (size_t j = 1; j != S; ++j)
                std::tie(c, t[j - 1]) = addmul(t[j], m, mod_[j], c);
            tmp = intx::addc(t[S], c);
            t[S - 1] = tmp.value;
            t[S] = d + tmp.carry;  // TODO: Carry is 0 for sparse modulus.
        }

        if (t >= mod_)
            t -= mod_;

        return static_cast<UintT>(t);
    }

    /// Performs a modular addition.
    constexpr UintT add(const UintT& x, const UintT& y) const noexcept
    {
#if defined(SP1) || defined(SP1TURBO)
        if constexpr (BN)
        {
            UintT res = x;
            syscall_bn254_fp_addmod(
                reinterpret_cast<size_t*>(&res), reinterpret_cast<const size_t*>(&y));
            return res;
        }
#endif

        // Using generic procedure is fine for Montgomery forms.
        return modadd(x, y, mod_);
    }

    /// Performs a modular subtraction.
    constexpr UintT sub(const UintT& x, const UintT& y) const noexcept
    {
#if defined(SP1) || defined(SP1TURBO)
        if constexpr (BN)
        {
            UintT res = x;
            syscall_bn254_fp_submod(
                reinterpret_cast<size_t*>(&res), reinterpret_cast<const size_t*>(&y));
            return res;
        }
#endif

        // Using generic procedure is fine for Montgomery forms.
        return modsub(x, y, mod_);
    }

    /// Computes modular inverse of x in Montgomery form. Result is in Montgomery form.
    /// Returns 0 for non-invertible x (including x == 0).
    constexpr UintT inv(const UintT& x) const noexcept
    {
        assert((mod_ & 1) == 1);
        assert(mod_ >= 3);
#if defined(__SIZEOF_INT128__)
        if constexpr (UintT::num_bits == 256)
        {
            // Variable-time safegcd, several times cheaper than the binary GCD below,
            // which remains the fallback for other widths and the test reference.
            // Seeding with R² inverts the Montgomery value directly, so neither a
            // from_mont() on the way in nor a to_mont() on the way out is needed.
            // This also covers the SP1 BN case, where R² is 1 and values are not in
            // Montgomery form: the seed is then the plain 1 the algorithm expects.
            return detail::modinv256(x, mod_, r_squared_);
        }
        else
#endif
        {
            return inv_binary_gcd(x);
        }
    }

    /// Reference modular inversion: binary extended GCD, constant path count.
    /// Input and result in Montgomery form; returns 0 for non-invertible x.
    constexpr UintT inv_binary_gcd(const UintT& x) const noexcept
    {
        assert((mod_ & 1) == 1);
        assert(mod_ >= 3);

        // The input XR would invert to X⁻¹R⁻¹, so scaling by R² gives the expected Montgomery
        // form X⁻¹R at no extra cost.
        return modinv_scaled(x, r_squared_, mod_);
    }
};
}  // namespace evmone::crypto
