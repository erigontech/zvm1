// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2023 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#include "secp256k1.hpp"
#include "keccak.hpp"

#if defined(SP1TURBO) || defined(SP1)
#include <sp1_syscalls.hpp>
#endif

namespace evmmax::secp256k1
{
namespace
{
constexpr auto B = Curve::Fp{7};

constexpr AffinePoint G{0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798_u256,
    0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8_u256};

/// Precomputed window-8 table: G_TABLE[i] = (i+1)*G for i=0..254.
/// 255 consteval AffinePoints = 16KB in .rodata.
/// Window-8 is the sweet spot: ~30 G-additions vs ~128 binary.
/// Larger windows (12, 15) cause memory access overhead that offsets savings.
constexpr int G_TABLE_SIZE = 255;
constexpr unsigned G_WINDOW = 8;
constexpr unsigned G_WINDOW_MASK = 0xFF;
// NOLINTNEXTLINE(*-avoid-c-arrays)
constexpr AffinePoint G_TABLE[G_TABLE_SIZE] = {
#include "secp256k1_g_table_w8.inc"
};

/// Window-8 MSM using precomputed G-table: computes u*G + v*R.
ecc::ProjPoint<Curve> msm_with_g_table(
    const uint256& u, const uint256& v, const AffinePoint& R) noexcept
{
    ecc::ProjPoint<Curve> result;

    const auto bit_width = intx::bit_width(u | v);
    if (bit_width == 0)
        return result;

    const auto aligned_width =
        ((bit_width + G_WINDOW - 1) / G_WINDOW) * G_WINDOW;

    for (auto i = aligned_width; i != 0; --i)
    {
        result = ecc::dbl(result);

        if (i <= bit_width && intx::bit_test(v, i - 1))
            result = ecc::add(result, R);

        if (((i - 1) % G_WINDOW) == 0 && i <= bit_width)
        {
            const auto shift = i - 1;
            const auto u_win = static_cast<unsigned>((u >> shift) & G_WINDOW_MASK);
            if (u_win != 0)
                result = ecc::add(result, G_TABLE[u_win - 1]);
        }
    }

    return result;
}

/// Precomputed phi(G) = (BETA * G.x, G.y) for GLV endomorphism.
/// phi(P) = (BETA*P.x, P.y) satisfies phi(P) = [LAMBDA]*P on secp256k1.
constexpr auto make_phi_g() noexcept
{
    const auto beta = Curve::Fp{Curve::BETA};
    return AffinePoint{beta * G.x, G.y};
}
constexpr AffinePoint PHI_G = make_phi_g();

/// GLV 4-way MSM: computes u1*G + u2*R using scalar decomposition and endomorphism.
///
/// Decomposes each 256-bit scalar into two ~128-bit half-scalars via the GLV lattice,
/// then runs a 4-way Shamir scan over ~128 bits instead of ~256 bits.
/// Uses batch inversion (Montgomery's trick) to build the 15-entry lookup table
/// with only 2 field inversions instead of 11.
ecc::ProjPoint<Curve> ecrecover_msm_glv(
    const uint256& u1, const uint256& u2, const AffinePoint& R) noexcept
{
    using FE = Curve::Fp;

    // 1. Decompose scalars: u1 = k1a + k1b*lambda, u2 = k2a + k2b*lambda
    auto [sk1a, sk1b] = ecc::decompose<Curve>(u1);
    auto [sk2a, sk2b] = ecc::decompose<Curve>(u2);

    // Compute phi(R) = (BETA * R.x, R.y)
    const FE beta{Curve::BETA};
    const AffinePoint phi_R{beta * R.x, R.y};

    // Handle signs: negate point if scalar is negative.
    AffinePoint P1 = sk1a.sign ? -G : G;
    AffinePoint P2 = sk1b.sign ? -PHI_G : PHI_G;
    AffinePoint P3 = sk2a.sign ? -R : R;
    AffinePoint P4 = sk2b.sign ? -phi_R : phi_R;

    const auto& k1a = sk1a.value;
    const auto& k1b = sk1b.value;
    const auto& k2a = sk2a.value;
    const auto& k2b = sk2b.value;

    // 2. Build 15-entry Shamir lookup table using batch inversion.
    //    Indexed by 4-bit mask: bit0=k1a, bit1=k1b, bit2=k2a, bit3=k2b.
    AffinePoint table[15];
    table[0]  = P1;  // 0001
    table[1]  = P2;  // 0010
    table[3]  = P3;  // 0100
    table[7]  = P4;  // 1000

    // Helper for batch inversion point addition.
    struct AddData { FE dx; FE dy; FE x1; FE y1; FE x2; };

    auto compute_add = [](const AddData& d, const FE& inv_dx) -> AffinePoint {
        const auto slope = d.dy * inv_dx;
        const auto xr = slope * slope - d.x1 - d.x2;
        const auto yr = slope * (d.x1 - xr) - d.y1;
        return {xr, yr};
    };

    // --- Batch 1: 6 independent pairwise additions ---
    AddData b1[6];
    b1[0] = {P2.x - P1.x, P2.y - P1.y, P1.x, P1.y, P2.x};  // P1+P2
    b1[1] = {P3.x - P1.x, P3.y - P1.y, P1.x, P1.y, P3.x};  // P1+P3
    b1[2] = {P3.x - P2.x, P3.y - P2.y, P2.x, P2.y, P3.x};  // P2+P3
    b1[3] = {P4.x - P1.x, P4.y - P1.y, P1.x, P1.y, P4.x};  // P1+P4
    b1[4] = {P4.x - P2.x, P4.y - P2.y, P2.x, P2.y, P4.x};  // P2+P4
    b1[5] = {P4.x - P3.x, P4.y - P3.y, P3.x, P3.y, P4.x};  // P3+P4

    FE acc1[6];
    acc1[0] = b1[0].dx;
    for (int i = 1; i < 6; ++i)
        acc1[i] = acc1[i - 1] * b1[i].dx;

    FE inv_a1 = 1 / acc1[5];

    FE i1[6];
    for (int i = 5; i > 0; --i)
    {
        i1[i] = inv_a1 * acc1[i - 1];
        inv_a1 = inv_a1 * b1[i].dx;
    }
    i1[0] = inv_a1;

    table[2]  = compute_add(b1[0], i1[0]); // P1+P2
    table[4]  = compute_add(b1[1], i1[1]); // P1+P3
    table[5]  = compute_add(b1[2], i1[2]); // P2+P3
    table[8]  = compute_add(b1[3], i1[3]); // P1+P4
    table[9]  = compute_add(b1[4], i1[4]); // P2+P4
    table[11] = compute_add(b1[5], i1[5]); // P3+P4

    // --- Batch 2: 5 additions depending on batch 1 results ---
    AddData b2[5];
    b2[0] = {P3.x - table[2].x, P3.y - table[2].y, table[2].x, table[2].y, P3.x};
    b2[1] = {P4.x - table[2].x, P4.y - table[2].y, table[2].x, table[2].y, P4.x};
    b2[2] = {P4.x - table[4].x, P4.y - table[4].y, table[4].x, table[4].y, P4.x};
    b2[3] = {P4.x - table[5].x, P4.y - table[5].y, table[5].x, table[5].y, P4.x};
    b2[4] = {table[5].x - table[8].x, table[5].y - table[8].y,
             table[8].x, table[8].y, table[5].x};

    FE acc2[5];
    acc2[0] = b2[0].dx;
    for (int i = 1; i < 5; ++i)
        acc2[i] = acc2[i - 1] * b2[i].dx;

    FE inv_a2 = 1 / acc2[4];

    FE i2[5];
    for (int i = 4; i > 0; --i)
    {
        i2[i] = inv_a2 * acc2[i - 1];
        inv_a2 = inv_a2 * b2[i].dx;
    }
    i2[0] = inv_a2;

    table[6]  = compute_add(b2[0], i2[0]); // (P1+P2)+P3
    table[10] = compute_add(b2[1], i2[1]); // (P1+P2)+P4
    table[12] = compute_add(b2[2], i2[2]); // (P1+P3)+P4
    table[13] = compute_add(b2[3], i2[3]); // (P2+P3)+P4
    table[14] = compute_add(b2[4], i2[4]); // (P1+P4)+(P2+P3) = P1+P2+P3+P4

    // 3. 4-way Shamir scan over ~128 bits.
    const auto bw = intx::bit_width(k1a | k1b | k2a | k2b);
    if (bw == 0)
        return {};

    ecc::ProjPoint<Curve> result;
    for (auto i = bw; i != 0; --i)
    {
        result = ecc::dbl(result);

        const unsigned idx =
            (unsigned{intx::bit_test(k1a, i - 1)} << 0) |
            (unsigned{intx::bit_test(k1b, i - 1)} << 1) |
            (unsigned{intx::bit_test(k2a, i - 1)} << 2) |
            (unsigned{intx::bit_test(k2b, i - 1)} << 3);
        if (idx != 0)
            result = ecc::add(result, table[idx - 1]);
    }

    return result;
}

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

}  // namespace
#endif


std::optional<AffinePoint> secp256k1_ecdsa_recover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes,
    bool parity) noexcept
{
    // Follows "Elliptic Curve Digital Signature Algorithm - Public key recovery"
    // https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm#Public_key_recovery

    // 1. Validate r and s are within [1, n-1].
    const auto opt_r = Curve::Fr::from_bytes(r_bytes);
    if (!opt_r.has_value() || *opt_r == 0) [[unlikely]]
        return std::nullopt;

    const auto opt_s = Curve::Fr::from_bytes(s_bytes);
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
    const auto Rpt = AffinePoint{r_mont, *y};
#if defined(AIRBENDER) && defined(__riscv)
    // Use GLV endomorphism for 4-way MSM over ~128-bit scalars
    const auto Q = ecrecover_msm_glv(u1.value(), u2.value(), Rpt);
#else
    const auto Q = msm(u1.value(), G, u2.value(), Rpt);
#endif

    // The public key mustn't be the point at infinity. This check is cheaper on a non-affine point.
    if (Q == 0) [[unlikely]]
        return std::nullopt;

    return to_affine(Q);
}

std::optional<evmc::address> ecrecover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes,
    bool parity) noexcept
{
#if defined(SP1TURBO) || defined(SP1)
    // Validate r and s.
    const auto opt_r = Curve::Fr::from_bytes(r_bytes);
    if (!opt_r.has_value() || *opt_r == 0)
        return std::nullopt;
    const auto opt_s = Curve::Fr::from_bytes(s_bytes);
    if (!opt_s.has_value() || *opt_s == 0)
        return std::nullopt;
    const auto& r_fr = *opt_r;
    const auto& s_fr = *opt_s;

    // Compute z, u1, u2 using FieldElement arithmetic.
    const auto z = Curve::Fr{intx::be::unsafe::load<uint256>(hash.data())};
    const auto r_inv = 1 / r_fr;
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
    sp1_msm(sp1_Q, u1, sp1_G, u2, sp1_R);

    if (is_zero(sp1_Q)) [[unlikely]]
        return std::nullopt;

    uint8_t serialized[64];
    sp1_point_to_bytes(serialized, sp1_Q);
    return to_address(serialized);
#else
    const auto pubkey = secp256k1_ecdsa_recover(hash, r_bytes, s_bytes, parity);
    if (!pubkey.has_value())
        return std::nullopt;

    return to_address(*pubkey);
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


}  // namespace evmmax::secp256k1
