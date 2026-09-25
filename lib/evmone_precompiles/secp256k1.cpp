// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2023 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#include "secp256k1.hpp"
#include "keccak.hpp"
#include <vector>

#if defined(SP1TURBO) || defined(SP1)
#include <sp1_syscalls.hpp>
#endif

namespace evmone::crypto::secp256k1
{
namespace
{
constexpr auto B = Curve::Fp{7};

constexpr AffinePoint G{0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798_u256,
    0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8_u256};
}  // namespace

// FIXME: Change to "uncompress_point".
std::optional<Curve::Fp> calculate_y(const Curve::Fp& x, bool y_parity) noexcept
{
    // Calculate y = √(x³ + 7).
    const auto xxx = x * x * x;
    const auto opt_y = field_sqrt(xxx + B);
    if (!opt_y.has_value())
        return std::nullopt;

    // Negate if different parity requested.
    const auto& y = *opt_y;
    const auto candidate_parity = (y.value() & 1) != 0;
    return (candidate_parity == y_parity) ? y : -y;
}

evmc::address to_address(std::span<const uint8_t, 64> pubkey) noexcept
{
    const auto hashed = ethash::keccak256(pubkey.data(), pubkey.size());
    evmc::address ret;
    std::copy_n(&hashed.bytes[12], sizeof(ret), ret.bytes);
    return ret;
}

#if defined(SP1TURBO) || defined(SP1)

inline void sp1_mulmod(uint256& result, const uint256& x, const uint256& y)
{
    // TODO(sp1): This can be further optimized by requiring the layout from the caller.
    uint256 args[2];
    auto& arg = args[0];
    auto& mod = args[1];
    mod = Curve::FIELD_PRIME;
    arg = y;
    result = x;
    sp1::mulmod(result, args);
}

// Manual modular addition: (x + y) mod p
inline uint256 sp1_addmod(const uint256& x, const uint256& y)
{
    auto sum = intx::addc(x, y);
    // If carry or sum >= p, subtract p
    if (sum.carry || sum.value >= Curve::FIELD_PRIME)
        return sum.value - Curve::FIELD_PRIME;
    return sum.value;
}

// Manual modular subtraction: (x - y) mod p
inline uint256 sp1_submod(const uint256& x, const uint256& y)
{
    auto diff = intx::subc(x, y);
    // If borrow, add p
    if (diff.carry)
        return diff.value + Curve::FIELD_PRIME;
    return diff.value;
}


std::optional<uint256> field_sqrt_sp1(const uint256& field, const uint256& x) noexcept
{
    uint256 z;
    uint256 t0;
    uint256 t1;
    uint256 t2;
    uint256 t3;

    // Step 1: z = x^0x2
    sp1_mulmod(z, x, x);

    // Step 2: z = x^0x3
    sp1_mulmod(z, x, z);

    // Step 4: t0 = x^0xc
    sp1_mulmod(t0, z, z);
    for (int i = 1; i < 2; ++i)
        sp1_mulmod(t0, t0, t0);

    // Step 5: t0 = x^0xf
    sp1_mulmod(t0, z, t0);

    // Step 6: t1 = x^0x1e
    sp1_mulmod(t1, t0, t0);

    // Step 7: t2 = x^0x1f
    sp1_mulmod(t2, x, t1);

    // Step 9: t1 = x^0x7c
    sp1_mulmod(t1, t2, t2);
    for (int i = 1; i < 2; ++i)
        sp1_mulmod(t1, t1, t1);

    // Step 10: t1 = x^0x7f
    sp1_mulmod(t1, z, t1);

    // Step 14: t3 = x^0x7f0
    sp1_mulmod(t3, t1, t1);
    for (int i = 1; i < 4; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 15: t0 = x^0x7ff
    sp1_mulmod(t0, t0, t3);

    // Step 26: t3 = x^0x3ff800
    sp1_mulmod(t3, t0, t0);
    for (int i = 1; i < 11; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 27: t0 = x^0x3fffff
    sp1_mulmod(t0, t0, t3);

    // Step 32: t3 = x^0x7ffffe0
    sp1_mulmod(t3, t0, t0);
    for (int i = 1; i < 5; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 33: t2 = x^0x7ffffff
    sp1_mulmod(t2, t2, t3);

    // Step 60: t3 = x^0x3ffffff8000000
    sp1_mulmod(t3, t2, t2);
    for (int i = 1; i < 27; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 61: t2 = x^0x3fffffffffffff
    sp1_mulmod(t2, t2, t3);

    // Step 115: t3 = x^0xfffffffffffffc0000000000000
    sp1_mulmod(t3, t2, t2);
    for (int i = 1; i < 54; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 116: t2 = x^0xfffffffffffffffffffffffffff
    sp1_mulmod(t2, t2, t3);

    // Step 224: t3 = x^0xfffffffffffffffffffffffffff000000000000000000000000000
    sp1_mulmod(t3, t2, t2);
    for (int i = 1; i < 108; ++i)
        sp1_mulmod(t3, t3, t3);

    // Step 225: t2 = x^0xffffffffffffffffffffffffffffffffffffffffffffffffffffff
    sp1_mulmod(t2, t2, t3);

    // Step 232: t2 = x^0x7fffffffffffffffffffffffffffffffffffffffffffffffffffff80
    for (int i = 0; i < 7; ++i)
        sp1_mulmod(t2, t2, t2);

    // Step 233: t1 = x^0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffff
    sp1_mulmod(t1, t1, t2);

    // Step 256: t1 = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffff800000
    for (int i = 0; i < 23; ++i)
        sp1_mulmod(t1, t1, t1);

    // Step 257: t0 = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff
    sp1_mulmod(t0, t0, t1);

    // Step 263: t0 = x^0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc0
    for (int i = 0; i < 6; ++i)
        sp1_mulmod(t0, t0, t0);

    // Step 264: z = x^0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc3
    sp1_mulmod(z, z, t0);

    // Step 266: z = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    for (int i = 0; i < 2; ++i)
        sp1_mulmod(z, z, z);

    // Verify: z^2 == x (mod p)
    uint256 z_squared;
    sp1_mulmod(z_squared, z, z);

    if (z_squared != x)
        return std::nullopt;  // Computed value is not the square root.

    return z;
}


/// Decompress a secp256k1 point from x-coordinate and y-parity (SP1 version).
/// Returns y-coordinate in regular (non-Montgomery) form, or nullopt if x is not on curve.
/// This version uses SP1 syscalls which operate on regular form values.

std::optional<uint256> decompress(const uint256& x, bool y_parity) noexcept
{
    // Calculate x^3 + 7 (all in regular/non-Montgomery form for SP1 syscalls)
    constexpr uint256 B_regular = 7;

    uint256 x_squared{};
    sp1_mulmod(x_squared, x, x);  // x_squared = x^2 mod p

    uint256 x_cubed{};
    sp1_mulmod(x_cubed, x_squared, x);  // x_cubed = x^3 mod p

    uint256 y_squared = sp1_addmod(x_cubed, B_regular);  // y_squared = x^3 + 7 mod p

    // sqrt(x^3 + 7)
    const auto y = field_sqrt_sp1(Curve::FIELD_PRIME, y_squared);
    if (!y.has_value())
        return std::nullopt;

    // Check parity and negate if needed (using regular form arithmetic)
    const auto candidate_parity = (*y & 1) != 0;
    if (candidate_parity == y_parity)
        return *y;
    else
        return sp1_submod(uint256{0}, *y);  // Return (0 - y) mod p
}
#endif


evmc::address to_address(const AffinePoint& pt) noexcept
{
    uint8_t serialized[64];
    pt.to_bytes(serialized);
    return to_address(serialized);
}

#if defined(SP1TURBO) || defined(SP1)
namespace
{
constexpr auto Gx_val = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798_u256;
constexpr auto Gy_val = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8_u256;

constexpr size_t SP1_POINT_SIZE = 64 / sizeof(size_t);

// sp1_G needs platform-specific constexpr initialization due to different limb sizes.
#ifdef SP1TURBO
constexpr sp1_AffinePoint sp1_G = {
    static_cast<uint32_t>(Gx_val[0]),
    static_cast<uint32_t>(Gx_val[0] >> 32),
    static_cast<uint32_t>(Gx_val[1]),
    static_cast<uint32_t>(Gx_val[1] >> 32),
    static_cast<uint32_t>(Gx_val[2]),
    static_cast<uint32_t>(Gx_val[2] >> 32),
    static_cast<uint32_t>(Gx_val[3]),
    static_cast<uint32_t>(Gx_val[3] >> 32),
    static_cast<uint32_t>(Gy_val[0]),
    static_cast<uint32_t>(Gy_val[0] >> 32),
    static_cast<uint32_t>(Gy_val[1]),
    static_cast<uint32_t>(Gy_val[1] >> 32),
    static_cast<uint32_t>(Gy_val[2]),
    static_cast<uint32_t>(Gy_val[2] >> 32),
    static_cast<uint32_t>(Gy_val[3]),
    static_cast<uint32_t>(Gy_val[3] >> 32),
};
#else  // SP1
constexpr sp1_AffinePoint sp1_G = {
    Gx_val[0], Gx_val[1], Gx_val[2], Gx_val[3],
    Gy_val[0], Gy_val[1], Gy_val[2], Gy_val[3],
};
#endif

/// Add non-zero p to non-zero r with first-limb fast reject on x-coordinate.
[[gnu::always_inline]] inline bool sp1_secp256k1_add_nz(sp1_AffinePoint r, const sp1_AffinePoint p) noexcept
{
    // Quick reject on first limb of x-coordinate (~1/2^64 false positive).
    if (r[0] != p[0]) [[likely]]
    {
        syscall_secp256k1_add(r, p);
        return true;
    }
    // Full x-coordinate comparison (only reached ~1/2^64 of the time).
    if (reinterpret_cast<const uint256&>(r[0]) != reinterpret_cast<const uint256&>(p[0])) [[likely]]
    {
        syscall_secp256k1_add(r, p);
        return true;
    }

    const auto& ry = reinterpret_cast<const uint256&>(r[SP1_POINT_SIZE / 2]);
    const auto& py = reinterpret_cast<const uint256&>(p[SP1_POINT_SIZE / 2]);
    if (ry == py)
    {
        syscall_secp256k1_double(r);
        return true;
    }
    else
    {  // r == -p
        std::fill_n(r, SP1_POINT_SIZE, 0);
        return false;  // r becomes zero
    }
}

/// Add p to r, handling edge cases that SP1 syscall doesn't support:
/// zero points (infinity) and points with the same x-coordinate.
void sp1_secp256k1_add(sp1_AffinePoint r, const sp1_AffinePoint p) noexcept
{
    if (is_zero(p)) [[unlikely]]
        return;
    if (is_zero(r)) [[unlikely]]
    {
        std::copy_n(p, SP1_POINT_SIZE, r);
        return;
    }

    sp1_secp256k1_add_nz(r, p);
}

/// SP1 version of ecc::msm() — computes multi-scalar multiplication u×P + v×Q
/// using SP1 syscalls for point operations.
/// See: ecc.hpp::msm(), https://eprint.iacr.org/2003/257.pdf#page=7.
void sp1_msm(sp1_AffinePoint r, const uint256& u, const sp1_AffinePoint p,
    const uint256& v, const sp1_AffinePoint q) noexcept
{
    // Precompute affine P + Q (safe add handles P==Q, P==-Q, and zero points).
    sp1_AffinePoint h;
    std::copy_n(p, SP1_POINT_SIZE, h);
    sp1_secp256k1_add(h, q);

    // Q == -P makes h the zero (infinity) encoding, which the raw syscalls in the
    // loops below reject. Then uP + vQ == (u-v)P, a single scalar multiplication.
    // Handled out of line with the identity-safe add rather than the _nz fast
    // path, so the hot loops keep exactly the shape they had before this case
    // was handled. Only reachable on adversarial input.
    if (is_zero(h)) [[unlikely]]
    {
        const uint64_t* base;
        uint256 d;
        if (u >= v)
        {
            d = u - v;
            base = p;
        }
        else
        {
            d = v - u;  // (u-v)P == (v-u)(-P) == (v-u)Q
            base = q;
        }
        std::fill_n(r, SP1_POINT_SIZE, 0);
        for (auto j = intx::bit_width(d); j != 0; --j)
        {
            if (!is_zero(r))
                syscall_secp256k1_double(r);
            if (intx::bit_test(d, j - 1))
                sp1_secp256k1_add(r, base);
        }
        return;
    }

    // Lookup table: index = (v_bit << 1) | u_bit.
    const uint64_t* points[4] = {nullptr, p, q, h};

    // Find the bit width across both scalars simultaneously.
    const auto bw = std::max(intx::bit_width(u), intx::bit_width(v));

    if (bw == 0)
    {
        std::fill_n(r, SP1_POINT_SIZE, 0);
        return;
    }

    // Find the first non-zero index to initialize the accumulator.
    size_t i = bw;
    for (; i != 0; --i)
    {
        const auto idx =
            2 * unsigned{intx::bit_test(v, i - 1)} + unsigned{intx::bit_test(u, i - 1)};
        if (idx != 0)
        {
            std::copy_n(points[idx], SP1_POINT_SIZE, r);
            --i;
            break;
        }
    }
    // Main loop: double-and-add. Zero-point checks skipped (accumulator and
    // table points are always non-zero), only same-x check via _nz helper.
    bool nz = true;
    for (; i != 0 && nz; --i)
    {
        syscall_secp256k1_double(r);
        const auto idx =
            2 * unsigned{intx::bit_test(v, i - 1)} + unsigned{intx::bit_test(u, i - 1)};
        if (idx != 0)
            nz = sp1_secp256k1_add_nz(r, points[idx]);
    }
    // If r becomes zero                                                             
    for (; i != 0; --i)
    {
        if (!is_zero(r))
            syscall_secp256k1_double(r);
        const auto idx =
            2 * unsigned{intx::bit_test(v, i - 1)} + unsigned{intx::bit_test(u, i - 1)};
        if (idx != 0)
            sp1_secp256k1_add(r, points[idx]);
    }
}


#ifndef SP1TURBO
// GLV endomorphism: phi(x, y) = (BETA * x, y) = LAMBDA * (x, y). A scalar splits as
// k = k1 + k2 * LAMBDA (mod N) with |k1|, |k2| < 2^128 (libsecp256k1 scalar_split_lambda),
// so u*G + v*Q becomes four half-length scalars sharing one run of 128 doublings.
constexpr auto GLV_BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee_u256;
constexpr auto GLV_G1 = 0x3086d221a7d46bcde86c90e49284eb153daa8a1471e8ca7fe893209a45dbb031_u256;
constexpr auto GLV_G2 = 0xe4437ed6010e88286f547fa90abfe4c4221208ac9df506c61571b4ae8ac47f71_u256;
constexpr auto GLV_MINUS_B1 = 0xe4437ed6010e88286f547fa90abfe4c3_u256;
constexpr auto GLV_B2 = 0x3086d221a7d46bcde86c90e49284eb15_u256;  // == A1
constexpr auto GLV_A2 = 0x114ca50f7a8e2f3f657c1108d9d44cfd8_u256;

constexpr auto PhiGx_val = 0xbcace2e99da01887ab0102b696902325872844067f15e98da7bba04400b88fcb_u256;
constexpr auto GpPhiGx_val = 0xc994b69768832bcbff5e9ab39ae8d1d3763bbf1e531bed98fe51de5ee84f50fb_u256;
constexpr auto GpPhiGy_val = 0xb7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777_u256;
constexpr auto GmPhiGx_val = 0x93c4d65b4cc437be9f2b0aa72325ba6ce5015022596e21f2ea6eadae415a87b0_u256;
constexpr auto GmPhiGy_val = 0xde87653b1778d37f77e9403692bd956f5d419b1b625309ca50730fbc03706352_u256;

constexpr sp1_AffinePoint sp1_PhiG = {PhiGx_val[0], PhiGx_val[1], PhiGx_val[2], PhiGx_val[3],
    Gy_val[0], Gy_val[1], Gy_val[2], Gy_val[3]};
constexpr sp1_AffinePoint sp1_GpPhiG = {GpPhiGx_val[0], GpPhiGx_val[1], GpPhiGx_val[2],
    GpPhiGx_val[3], GpPhiGy_val[0], GpPhiGy_val[1], GpPhiGy_val[2], GpPhiGy_val[3]};
constexpr sp1_AffinePoint sp1_GmPhiG = {GmPhiGx_val[0], GmPhiGx_val[1], GmPhiGx_val[2],
    GmPhiGx_val[3], GmPhiGy_val[0], GmPhiGy_val[1], GmPhiGy_val[2], GmPhiGy_val[3]};

/// A GLV half-scalar: magnitude < 2^128 and sign.
struct GlvHalf
{
    uint64_t lo;
    uint64_t hi;
    bool neg;
};

/// round(k * g / 2^384).
inline uint256 mul_shift_384(const uint256& k, const uint256& g) noexcept
{
    const auto p = intx::umul(k, g);
    return uint256{p[6], p[7], 0, 0} + (p[5] >> 63);
}

inline GlvHalf glv_half(const uint256& t) noexcept
{
    const bool neg = (t[3] >> 63) != 0;
    const auto m = neg ? uint256{0} - t : t;
    return {m[0], m[1], neg};
}

inline void glv_split(const uint256& k, GlvHalf& k1, GlvHalf& k2) noexcept
{
    const auto c1 = mul_shift_384(k, GLV_G1);
    const auto c2 = mul_shift_384(k, GLV_G2);
    // Exact small values, computed modulo 2^256.
    k2 = glv_half(c1 * GLV_MINUS_B1 - c2 * GLV_B2);
    k1 = glv_half(k - c1 * GLV_B2 - c2 * GLV_A2);
}

/// r = -r: y = P - y (no point of secp256k1 has y == 0).
inline void sp1_negate(sp1_AffinePoint r) noexcept
{
    auto& y = reinterpret_cast<uint256&>(r[SP1_POINT_SIZE / 2]);
    y = Curve::FIELD_PRIME - y;
}

/// Spreads the 16 bits of x to bits 0, 4, 8, ..., 60.
inline uint64_t spread4(uint64_t x) noexcept
{
    x &= 0xffff;
    x = (x | (x << 24)) & 0x000000ff000000ff;
    x = (x | (x << 12)) & 0x000f000f000f000f;
    x = (x | (x << 6)) & 0x0303030303030303;
    x = (x | (x << 3)) & 0x1111111111111111;
    return x;
}

/// SP1 u×G + v×Q with the GLV endomorphism. Falls back to sp1_msm() when a table entry is the
/// point at infinity, which only an adversarial Q reaches.
void sp1_msm_glv(sp1_AffinePoint r, const uint256& u, const uint256& v, const sp1_AffinePoint q) noexcept
{
    GlvHalf k[4];
    glv_split(u, k[0], k[1]);
    glv_split(v, k[2], k[3]);

    // t[i] is the sum of the signed bases selected by the bits of i: G, phi(G), Q, phi(Q).
    sp1_AffinePoint t[16];
    std::copy_n(sp1_G, SP1_POINT_SIZE, t[1]);
    std::copy_n(sp1_PhiG, SP1_POINT_SIZE, t[2]);
    std::copy_n(k[0].neg == k[1].neg ? sp1_GpPhiG : sp1_GmPhiG, SP1_POINT_SIZE, t[3]);
    if (k[0].neg)
    {
        sp1_negate(t[1]);
        sp1_negate(t[3]);  // -(G + phi(G)) or -(G - phi(G))
    }
    if (k[1].neg)
        sp1_negate(t[2]);
    std::copy_n(q, SP1_POINT_SIZE, t[4]);
    if (k[2].neg)
        sp1_negate(t[4]);
    sp1_mulmod(reinterpret_cast<uint256&>(t[8][0]), GLV_BETA, reinterpret_cast<const uint256&>(q[0]));
    std::copy_n(&q[SP1_POINT_SIZE / 2], SP1_POINT_SIZE / 2, &t[8][SP1_POINT_SIZE / 2]);
    if (k[3].neg)
        sp1_negate(t[8]);
    std::copy_n(t[4], SP1_POINT_SIZE, t[12]);
    sp1_secp256k1_add(t[12], t[8]);
    for (const unsigned hi : {4u, 8u, 12u})
    {
        for (const unsigned lo : {1u, 2u, 3u})
        {
            std::copy_n(t[lo], SP1_POINT_SIZE, t[hi | lo]);
            sp1_secp256k1_add(t[hi | lo], t[hi]);
        }
    }
    for (unsigned i = 4; i < 16; ++i)
    {
        if (is_zero(t[i])) [[unlikely]]
            return sp1_msm(r, u, sp1_G, v, q);
    }

    // Digits d = bit(k0) | bit(k1) << 1 | bit(k2) << 2 | bit(k3) << 3, 16 per word, bit 16w + j
    // at nibble j of word w.
    uint64_t d[8];
    for (unsigned w = 0; w < 8; ++w)
    {
        const unsigned sh = (w % 4) * 16;
        const bool high = w >= 4;
        uint64_t x = 0;
        for (unsigned j = 0; j < 4; ++j)
            x |= spread4((high ? k[j].hi : k[j].lo) >> sh) << j;
        d[w] = x;
    }

    int w = 7;
    while (w >= 0 && d[w] == 0)
        --w;
    if (w < 0)
    {
        std::fill_n(r, SP1_POINT_SIZE, 0);
        return;
    }
    uint64_t dw = d[w];
    unsigned left = 16;
    while ((dw >> 60) == 0)
    {
        dw <<= 4;
        --left;
    }
    std::copy_n(t[dw >> 60], SP1_POINT_SIZE, r);
    dw <<= 4;
    --left;

    bool nz = true;
    for (;;)
    {
        for (; left != 0; --left)
        {
            const auto idx = static_cast<unsigned>(dw >> 60);
            dw <<= 4;
            if (nz) [[likely]]
            {
                syscall_secp256k1_double(r);
                if (idx != 0)
                    nz = sp1_secp256k1_add_nz(r, t[idx]);
            }
            else
            {
                // The accumulator hit infinity: identity-safe operations until it leaves it.
                if (idx != 0)
                    sp1_secp256k1_add(r, t[idx]);
                nz = !is_zero(r);
            }
        }
        if (--w < 0)
            break;
        dw = d[w];
        left = 16;
    }
}
#endif
}  // namespace
#endif


std::optional<AffinePoint> secp256k1_ecdsa_recover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes, bool parity,
    RecoveryMode mode) noexcept
{
    // Follows "Elliptic Curve Digital Signature Algorithm - Public key recovery"
    // https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm#Public_key_recovery

    // 1. Validate r and s are within [1, n-1].
    const auto opt_r = Curve::Fr::from_bytes(r_bytes);
    if (!opt_r.has_value() || *opt_r == 0) [[unlikely]]
        return std::nullopt;

    const auto opt_s = mode == RecoveryMode::strict ?
                           Curve::Fr::from_bytes<Curve::Fr::Range::half>(s_bytes) :
                           Curve::Fr::from_bytes<Curve::Fr::Range::full>(s_bytes);
    if (!opt_s.has_value() || *opt_s == 0) [[unlikely]]
        return std::nullopt;

    const auto& r = *opt_r;
    const auto& s = *opt_s;

    // 3. Hash of the message is already calculated in e.
    // 4. Convert hash e to z field element by doing z = e % n.
    //    https://www.rfc-editor.org/rfc/rfc6979#section-2.3.2
    //    Converting to Montgomery form performs the e % n reduction.
    const auto z = Curve::Fr{intx::be::unsafe::load<uint256>(hash.data())};

    // 5. Calculate u1 and u2.
    const auto r_inv = 1 / r;
    const auto u1 = -z * r_inv;
    const auto u2 = s * r_inv;
    assert(u2 != 0);  // Because s != 0 and r_inv != 0.

    // 2. Calculate y coordinate of R from r and v.
    const auto r_mont = Curve::Fp{r.value()};
    const auto y = calculate_y(r_mont, parity);
    if (!y.has_value())
        return std::nullopt;

    // 6. Calculate public key point Q = u1×G + u2×R.
    const auto R = AffinePoint{r_mont, *y};
    const auto Q = msm(u1.value(), G, u2.value(), R);

    // The public key mustn't be the point at infinity. This check is cheaper on a non-affine point.
    if (Q == 0) [[unlikely]]
        return std::nullopt;

    return to_affine(Q);
}

#if defined(SP1TURBO) || defined(SP1)
namespace
{
/// Parses and range-checks the signature scalars as ecrecover() does.
bool parse_signature(std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes,
    RecoveryMode mode, Curve::Fr& r, Curve::Fr& s) noexcept
{
    const auto opt_r = Curve::Fr::from_bytes(r_bytes);
    if (!opt_r.has_value() || *opt_r == 0)
        return false;
    const auto opt_s = mode == RecoveryMode::strict ?
                           Curve::Fr::from_bytes<Curve::Fr::Range::half>(s_bytes) :
                           Curve::Fr::from_bytes<Curve::Fr::Range::full>(s_bytes);
    if (!opt_s.has_value() || *opt_s == 0)
        return false;
    r = *opt_r;
    s = *opt_s;
    return true;
}

/// ecrecover() after the scalar inversion, given r_inv = 1/r.
std::optional<evmc::address> ecrecover_inverted(std::span<const uint8_t, 32> hash,
    const Curve::Fr& r_fr, const Curve::Fr& s_fr, const Curve::Fr& r_inv, bool parity) noexcept
{
    // Compute z, u1, u2 using FieldElement arithmetic.
    const auto z = Curve::Fr{intx::be::unsafe::load<uint256>(hash.data())};
    const auto u1 = (-z * r_inv).value();
    const auto u2 = (s_fr * r_inv).value();
    assert(u2 != 0);

    const auto r_val = r_fr.value();

    // Point decompression and SP1 point operations.
#ifdef SP1TURBO
    uint8_t sp1_Rbytes[64]{};
    intx::be::unsafe::store(&sp1_Rbytes[0], r_val);
    syscall_secp256k1_decompress(sp1_Rbytes, parity);
    const auto y_sp1 = intx::be::unsafe::load<uint256>(&sp1_Rbytes[32]);
    if (y_sp1 == 0)
        return std::nullopt;
#else
    const auto y = decompress(r_val, parity);
    if (!y.has_value())
        return std::nullopt;
    uint8_t sp1_Rbytes[64]{};
    intx::be::unsafe::store(&sp1_Rbytes[0], r_val);
    intx::be::unsafe::store(&sp1_Rbytes[32], *y);
#endif

    sp1_AffinePoint sp1_R;
    sp1_point_from_bytes(sp1_R, sp1_Rbytes);

    // Shamir's trick: compute u1*G + u2*R in a single pass
    sp1_AffinePoint sp1_Q;
#ifdef SP1TURBO
    sp1_msm(sp1_Q, u1, sp1_G, u2, sp1_R);
#else
    sp1_msm_glv(sp1_Q, u1, u2, sp1_R);
#endif

    if (is_zero(sp1_Q)) [[unlikely]]
        return std::nullopt;

    uint8_t serialized[64];
    sp1_point_to_bytes(serialized, sp1_Q);
    return to_address(serialized);
}
}  // namespace
#endif

std::optional<evmc::address> ecrecover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes, bool parity,
    RecoveryMode mode) noexcept
{
#if defined(SP1TURBO) || defined(SP1)
    Curve::Fr r_fr, s_fr;
    if (!parse_signature(r_bytes, s_bytes, mode, r_fr, s_fr))
        return std::nullopt;
    return ecrecover_inverted(hash, r_fr, s_fr, 1 / r_fr, parity);
#else
    // TODO(C++23): use std::optional::and_then.
    const auto pubkey = secp256k1_ecdsa_recover(hash, r_bytes, s_bytes, parity, mode);
    if (!pubkey.has_value())
        return std::nullopt;

    return to_address(*pubkey);
#endif
}

void ecrecover_batch(std::span<const EcrecoverInput> in,
    std::span<std::optional<evmc::address>> out, RecoveryMode mode) noexcept
{
    assert(out.size() == in.size());
#if defined(SP1TURBO) || defined(SP1)
    // Montgomery's trick: prefix[i] = product of the valid r before i; invert the full product
    // once and peel each r^-1 off it walking back.
    const auto n = in.size();
    std::vector<Curve::Fr> r(n), s(n), prefix(n);
    std::vector<uint8_t> valid(n);
    auto prod = Curve::Fr::one();
    for (size_t i = 0; i < n; ++i)
    {
        valid[i] = parse_signature(in[i].r, in[i].s, mode, r[i], s[i]);
        if (valid[i])
        {
            prefix[i] = prod;
            prod = prod * r[i];
        }
    }
    auto inv = 1 / prod;  // r in [1, N) and N prime: prod != 0.
    for (size_t i = n; i-- > 0;)
    {
        if (!valid[i])
        {
            out[i] = std::nullopt;
            continue;
        }
        const auto r_inv = inv * prefix[i];
        inv = inv * r[i];
        out[i] = ecrecover_inverted(in[i].hash, r[i], s[i], r_inv, in[i].parity);
    }
#else
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = ecrecover(in[i].hash, in[i].r, in[i].s, in[i].parity, mode);
#endif
}

std::optional<Curve::Fp> field_sqrt(const Curve::Fp& x) noexcept
{
    // Computes modular exponentiation
    // x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    // Operations: 253 squares 13 multiplies
    // Main part generated by github.com/mmcloughlin/addchain v0.4.0.
    //   addchain search 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    //     > secp256k1_sqrt.acc
    //   addchain gen -tmpl expmod.tmpl secp256k1_sqrt.acc
    //     > secp256k1_sqrt.cpp
    //
    // Exponentiation computation is derived from the addition chain:
    //
    // _10      = 2*1
    // _11      = 1 + _10
    // _1100    = _11 << 2
    // _1111    = _11 + _1100
    // _11110   = 2*_1111
    // _11111   = 1 + _11110
    // _1111100 = _11111 << 2
    // _1111111 = _11 + _1111100
    // x11      = _1111111 << 4 + _1111
    // x22      = x11 << 11 + x11
    // x27      = x22 << 5 + _11111
    // x54      = x27 << 27 + x27
    // x108     = x54 << 54 + x54
    // x216     = x108 << 108 + x108
    // x223     = x216 << 7 + _1111111
    // return     ((x223 << 23 + x22) << 6 + _11) << 2

    // Allocate Temporaries.
    Curve::Fp z;
    Curve::Fp t0;
    Curve::Fp t1;
    Curve::Fp t2;
    Curve::Fp t3;


    // Step 1: z = x^0x2
    z = x * x;

    // Step 2: z = x^0x3
    z = x * z;

    // Step 4: t0 = x^0xc
    t0 = z * z;
    for (int i = 1; i < 2; ++i)
        t0 = t0 * t0;

    // Step 5: t0 = x^0xf
    t0 = z * t0;

    // Step 6: t1 = x^0x1e
    t1 = t0 * t0;

    // Step 7: t2 = x^0x1f
    t2 = x * t1;

    // Step 9: t1 = x^0x7c
    t1 = t2 * t2;
    for (int i = 1; i < 2; ++i)
        t1 = t1 * t1;

    // Step 10: t1 = x^0x7f
    t1 = z * t1;

    // Step 14: t3 = x^0x7f0
    t3 = t1 * t1;
    for (int i = 1; i < 4; ++i)
        t3 = t3 * t3;

    // Step 15: t0 = x^0x7ff
    t0 = t0 * t3;

    // Step 26: t3 = x^0x3ff800
    t3 = t0 * t0;
    for (int i = 1; i < 11; ++i)
        t3 = t3 * t3;

    // Step 27: t0 = x^0x3fffff
    t0 = t0 * t3;

    // Step 32: t3 = x^0x7ffffe0
    t3 = t0 * t0;
    for (int i = 1; i < 5; ++i)
        t3 = t3 * t3;

    // Step 33: t2 = x^0x7ffffff
    t2 = t2 * t3;

    // Step 60: t3 = x^0x3ffffff8000000
    t3 = t2 * t2;
    for (int i = 1; i < 27; ++i)
        t3 = t3 * t3;

    // Step 61: t2 = x^0x3fffffffffffff
    t2 = t2 * t3;

    // Step 115: t3 = x^0xfffffffffffffc0000000000000
    t3 = t2 * t2;
    for (int i = 1; i < 54; ++i)
        t3 = t3 * t3;

    // Step 116: t2 = x^0xfffffffffffffffffffffffffff
    t2 = t2 * t3;

    // Step 224: t3 = x^0xfffffffffffffffffffffffffff000000000000000000000000000
    t3 = t2 * t2;
    for (int i = 1; i < 108; ++i)
        t3 = t3 * t3;

    // Step 225: t2 = x^0xffffffffffffffffffffffffffffffffffffffffffffffffffffff
    t2 = t2 * t3;

    // Step 232: t2 = x^0x7fffffffffffffffffffffffffffffffffffffffffffffffffffff80
    for (int i = 0; i < 7; ++i)
        t2 = t2 * t2;

    // Step 233: t1 = x^0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffff
    t1 = t1 * t2;

    // Step 256: t1 = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffff800000
    for (int i = 0; i < 23; ++i)
        t1 = t1 * t1;

    // Step 257: t0 = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff
    t0 = t0 * t1;

    // Step 263: t0 = x^0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc0
    for (int i = 0; i < 6; ++i)
        t0 = t0 * t0;

    // Step 264: z = x^0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc3
    z = z * t0;

    // Step 266: z = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    for (int i = 0; i < 2; ++i)
        z = z * z;

    if (z * z != x)
        return std::nullopt;  // Computed value is not the square root.

    return z;
}
}  // namespace evmone::crypto::secp256k1
