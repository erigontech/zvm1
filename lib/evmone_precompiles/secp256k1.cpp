// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2023 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#include "secp256k1.hpp"
#include "keccak.hpp"
#include <memory>
#include <new>
#include <type_traits>

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

/// Precomputed window-8 table: PHI_G_TABLE[i] = (i+1)*phi(G) for i=0..254.
/// 255 consteval AffinePoints = 16KB in .rodata.
// NOLINTNEXTLINE(*-avoid-c-arrays)
constexpr AffinePoint PHI_G_TABLE[G_TABLE_SIZE] = {
#include "secp256k1_phig_table_w8.inc"
};

/// Hybrid GLV MSM: computes u1*G + u2*R using precomputed tables for G and phi(G).
///
/// Decomposes each 256-bit scalar into two ~128-bit half-scalars via the GLV lattice.
/// For the G component (u1a*G + u1b*phi(G)), uses precomputed window-8 tables (zero inversions).
/// For the R component (u2a*R + u2b*phi(R)), uses 2-way Shamir with a 3-entry table (1 inversion).
/// This saves ~2 field inversions and ~30 field muls vs the previous 15-entry combined table.
ecc::ProjPoint<Curve> ecrecover_msm_glv(
    const uint256& u1, const uint256& u2, const AffinePoint& R) noexcept
{
    using FE = Curve::Fp;

    // 1. Decompose scalars: u1 = k1a + k1b*lambda, u2 = k2a + k2b*lambda
    auto [sk1a, sk1b] = ecc::decompose<Curve>(u1);
    auto [sk2a, sk2b] = ecc::decompose<Curve>(u2);

    // 2. Build R-Shamir table for NAF-based Shamir.
    // Apply signs from decomposition to get P_a = +/-R, P_b = +/-phi(R).
    const AffinePoint phi_R{FE{Curve::BETA} * R.x, R.y};
    const AffinePoint P_a = sk2a.sign ? -R : R;
    const AffinePoint P_b = sk2b.sign ? -phi_R : phi_R;
    const AffinePoint neg_P_a = -P_a;
    const AffinePoint neg_P_b = -P_b;
    // P_a+P_b and P_a-P_b in Jacobian (no inversions needed).
    const auto P_sum = ecc::add(ecc::ProjPoint<Curve>(P_a), P_b);     // P_a + P_b
    const auto P_diff = ecc::add(ecc::ProjPoint<Curve>(P_a), neg_P_b);   // P_a - P_b
    // Precompute negations to avoid constructing temporaries in the hot loop.
    const auto neg_P_sum = -P_sum;
    const auto neg_P_diff = -P_diff;

    // 3. Signs for G/phi(G) table lookups -- negation applied per-lookup
    const bool g_neg = sk1a.sign;
    const bool phig_neg = sk1b.sign;

    const auto& k1a = sk1a.value;
    const auto& k1b = sk1b.value;
    const auto& k2a = sk2a.value;
    const auto& k2b = sk2b.value;

    // 4. Compute NAF for k2a and k2b, then build joint index array.
    // NAF digits are in {-1, 0, 1}, encoded as signed int8_t.
    // Joint index encodes (digit_a, digit_b) as a single byte:
    //   high nibble = digit_b + 1, low nibble = digit_a + 1
    //   so 0x00 = (-1,-1), 0x11 = (0,0), 0x22 = (1,1), etc.
    // But for speed, we use a flat encoding: 3*da + db (shifted by +4 to avoid negatives)
    // index = (da+1)*3 + (db+1) gives values 0..8 for the 9 combinations.
    // 0=(−1,−1) 1=(−1,0) 2=(−1,1) 3=(0,−1) 4=(0,0) 5=(0,1) 6=(1,−1) 7=(1,0) 8=(1,1)

    const auto bw = intx::bit_width(k1a | k1b | k2a | k2b);
    if (bw == 0)
        return {};

    // Compute NAF for k2a and k2b and build joint index array.
    // NAF digit rule: if scalar is odd, digit = 2 - (scalar % 4), then scalar -= digit.
    // Then scalar >>= 1. Uses 32-bit word-level operations for efficiency on RV32.
    uint8_t r_naf_idx[130];
    unsigned naf_len = 0;
    {
        // Work with 32-bit words directly for 128-bit scalars.
        uint32_t aw[4], bww[4];
        {
            const auto* ka = reinterpret_cast<const uint32_t*>(&k2a);
            const auto* kb = reinterpret_cast<const uint32_t*>(&k2b);
            for (int j = 0; j < 4; ++j) { aw[j] = ka[j]; bww[j] = kb[j]; }
        }

        auto is_zero4 = [](const uint32_t* w) {
            return (w[0] | w[1] | w[2] | w[3]) == 0;
        };
        auto shr1 = [](uint32_t* w) {
            w[0] = (w[0] >> 1) | (w[1] << 31);
            w[1] = (w[1] >> 1) | (w[2] << 31);
            w[2] = (w[2] >> 1) | (w[3] << 31);
            w[3] = w[3] >> 1;
        };
        // Add/sub a small value d to a 128-bit number (d is 1 or 2).
        // Native 32-bit arithmetic — avoids 64-bit emulation on rv32im.
        auto add_small = [](uint32_t* w, uint32_t d) {
            uint32_t sum = w[0] + d;
            uint32_t c = (sum < w[0]) ? 1u : 0u;
            w[0] = sum;
            for (int j = 1; j < 4 && c; ++j) {
                sum = w[j] + c;
                c = (sum < w[j]) ? 1u : 0u;
                w[j] = sum;
            }
        };
        auto sub_small = [](uint32_t* w, uint32_t d) {
            uint32_t c = (w[0] < d) ? 1u : 0u;
            w[0] -= d;
            for (int j = 1; j < 4 && c; ++j) {
                uint32_t prev = w[j];
                w[j] -= c;
                c = (prev < c) ? 1u : 0u;
            }
        };

        while (!is_zero4(aw) || !is_zero4(bww))
        {
            int8_t da = 0, db = 0;
            if (aw[0] & 1)
            {
                da = static_cast<int8_t>(2 - static_cast<int>(aw[0] & 3));
                if (da > 0) sub_small(aw, static_cast<uint32_t>(da));
                else add_small(aw, static_cast<uint32_t>(-da));
            }
            if (bww[0] & 1)
            {
                db = static_cast<int8_t>(2 - static_cast<int>(bww[0] & 3));
                if (db > 0) sub_small(bww, static_cast<uint32_t>(db));
                else add_small(bww, static_cast<uint32_t>(-db));
            }
            r_naf_idx[naf_len] = static_cast<uint8_t>((da + 1) * 3 + (db + 1));
            ++naf_len;
            shr1(aw);
            shr1(bww);
        }
    }

    // Precompute G-table window values (8-bit windows from k1a and k1b).
    uint8_t g_wins[32];
    {
        const auto* k1a_bytes = reinterpret_cast<const uint8_t*>(&k1a);
        const auto* k1b_bytes = reinterpret_cast<const uint8_t*>(&k1b);
        for (unsigned w = 0; w < 16; ++w)
        {
            g_wins[w] = k1a_bytes[w];
            g_wins[w + 16] = k1b_bytes[w];
        }
    }

    ecc::ProjPoint<Curve> result;

    // Determine the effective bit width including NAF extension.
    // Start from effective_bw (not aligned_bw) to skip wasted identity doublings.
    // The G-window lookups still fire at the right positions: (i-1) % 8 == 0.
    const auto effective_bw = std::max(static_cast<unsigned>(bw), naf_len);

    for (auto i = effective_bw; i != 0; --i)
    {
        result = ecc::dbl(result);

        // R-component: NAF-based Shamir with signed digits.
        // Note: (i-1) < naf_len implies i <= naf_len <= effective_bw.
        if ((i - 1) < naf_len)
        {
            const auto idx = r_naf_idx[i - 1];
            // idx encodes (da+1)*3 + (db+1); 4 = (0,0) = no-op
            // We decode da and db and use the lookup table.
            // idx: 0=(-1,-1) 1=(-1,0) 2=(-1,1) 3=(0,-1) 4=(0,0) 5=(0,1) 6=(1,-1) 7=(1,0) 8=(1,1)
            switch (idx)
            {
            case 4: break;  // (0,0): no addition
            case 7: result = ecc::add(result, P_a); break;       // (1,0): +P_a
            case 1: result = ecc::add(result, neg_P_a); break;   // (-1,0): -P_a
            case 5: result = ecc::add(result, P_b); break;       // (0,1): +P_b
            case 3: result = ecc::add(result, neg_P_b); break;   // (0,-1): -P_b
            case 8: result = ecc::add(result, P_sum); break;     // (1,1): +P_sum (Jac+Jac)
            case 0: result = ecc::add(result, neg_P_sum); break;   // (-1,-1): -P_sum
            case 6: result = ecc::add(result, P_diff); break;    // (1,-1): +P_diff (Jac+Jac)
            case 2: result = ecc::add(result, neg_P_diff); break; // (-1,1): -P_diff
            }
        }

        // G-component: window-8 lookup (every 8th bit, for 128-bit scalars = 16 windows).
        if (((i - 1) & (G_WINDOW - 1)) == 0 && i <= 128)
        {
            const auto win_idx = (i - 1) >> 3;  // (i - 1) / 8

            // u1a window -> G_TABLE
            {
                const auto u1a_win = static_cast<unsigned>(g_wins[win_idx]);
                if (u1a_win != 0)
                {
                    const auto& pt = G_TABLE[u1a_win - 1];
                    if (g_neg)
                        result = ecc::add(result, AffinePoint{pt.x, -pt.y});
                    else
                        result = ecc::add(result, pt);
                }
            }

            // u1b window -> PHI_G_TABLE
            {
                const auto u1b_win = static_cast<unsigned>(g_wins[win_idx + 16]);
                if (u1b_win != 0)
                {
                    const auto& pt = PHI_G_TABLE[u1b_win - 1];
                    if (phig_neg)
                        result = ecc::add(result, AffinePoint{pt.x, -pt.y});
                    else
                        result = ecc::add(result, pt);
                }
            }
        }
    }

    return result;
}

}  // namespace

// FIXME: Change to "uncompress_point".
__attribute__((flatten))
std::optional<Curve::Fp> calculate_y(const Curve::Fp& x, bool y_parity) noexcept
{
    // Calculate y = √(x³ + 7).
    auto xxx = x; xxx *= x;    // x^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
    xxx *= x;                   // x^3 (in-place, saves 1 MEMCOPY vs x * x * x)
    xxx += B;                   // x^3 + B (in-place, saves 1 MEMCOPY)
    const auto opt_y = field_sqrt(xxx);
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

}  // namespace
#endif


#if defined(AIRBENDER) && defined(__riscv)
namespace
{
ecc::ProjPoint<Curve> ecrecover_msm_single(
    const uint256& u1, const uint256& u2, const AffinePoint& R) noexcept;
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
    const auto Rpt = AffinePoint{r_mont, *y};
#if defined(AIRBENDER) && defined(__riscv)
    // GLV halves: width-12 NAFs over the precomputed odd multiples of G and phi(G), width-5 NAFs
    // over odd multiples of R and phi(R) in Jacobian coordinates.
    const auto Q = ecrecover_msm_single(u1.value(), u2.value(), Rpt);
#else
    const auto Q = msm(u1.value(), G, u2.value(), Rpt);
#endif

    // The public key mustn't be the point at infinity. This check is cheaper on a non-affine point.
    if (Q == 0) [[unlikely]]
        return std::nullopt;

    return to_affine(Q);
}

std::optional<evmc::address> ecrecover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes, bool parity,
    RecoveryMode mode) noexcept
{
#if defined(SP1TURBO) || defined(SP1)
    // Validate r and s.
    const auto opt_r = Curve::Fr::from_bytes(r_bytes);
    if (!opt_r.has_value() || *opt_r == 0)
        return std::nullopt;
    const auto opt_s = mode == RecoveryMode::strict ?
                           Curve::Fr::from_bytes<Curve::Fr::Range::half>(s_bytes) :
                           Curve::Fr::from_bytes<Curve::Fr::Range::full>(s_bytes);
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
    // TODO(C++23): use std::optional::and_then.
    const auto pubkey = secp256k1_ecdsa_recover(hash, r_bytes, s_bytes, parity, mode);
    if (!pubkey.has_value())
        return std::nullopt;

    return to_address(*pubkey);
#endif
}

#if defined(AIRBENDER) && defined(__riscv)
namespace
{
/// n default-constructed elements in 32-byte aligned storage: the BigInt CSR paths of the field
/// arithmetic need aligned operands, and the guest's allocator only guarantees 8 bytes.
template <typename T>
class AlignedArray
{
    static_assert(std::is_trivially_destructible_v<T>);
    std::unique_ptr<std::byte[]> raw_;
    T* p_;

public:
    explicit AlignedArray(size_t n) : raw_{new std::byte[n * sizeof(T) + 32]}
    {
        p_ = reinterpret_cast<T*>(
            (reinterpret_cast<uintptr_t>(raw_.get()) + 31) & ~static_cast<uintptr_t>(31));
        for (size_t i = 0; i < n; ++i)
            new (&p_[i]) T{};
    }
    T& operator[](size_t i) noexcept { return p_[i]; }
};

/// Replaces v[i] by its inverse for every i with live[i], using one field inversion
/// (Montgomery's trick). The live elements must be non-zero; prefix is scratch of the same size.
template <typename FE>
void batch_invert(AlignedArray<FE>& v, AlignedArray<FE>& prefix, const uint8_t* live, size_t n)
{
    size_t last = n;
    for (size_t i = 0; i < n; ++i)
    {
        if (!live[i])
            continue;
        prefix[i] = v[i];
        if (last != n)
            prefix[i] *= prefix[last];
        last = i;
    }
    if (last == n)
        return;
    auto inv = 1 / prefix[last];  // The inverse of the product of all live elements.
    for (size_t i = last;;)
    {
        size_t j = i;
        bool has_prev = false;
        while (j != 0)
        {
            if (live[--j])
            {
                has_prev = true;
                break;
            }
        }
        if (!has_prev)
        {
            v[i] = inv;
            return;
        }
        const auto vi = v[i];
        v[i] = inv;
        v[i] *= prefix[j];  // (v_0..v_i)^-1 * (v_0..v_j) = v_i^-1
        inv *= vi;          // Now (v_0..v_j)^-1.
        i = j;
    }
}

/// The signed digit width for the R half of the batched MSM, and its table of odd multiples
/// 1R, 3R, ..., 15R.
constexpr unsigned R_WNAF_W = 5;
constexpr size_t R_TABLE_SIZE = size_t{1} << (R_WNAF_W - 2);
/// The signed digit width for the G half: the precomputed odd multiples (2j+1)G and (2j+1)phi(G)
/// for j < 1024, so 128/13 additions per 128-bit half where window-8 lookups take 16.
constexpr unsigned G_WNAF_W = 12;
// NOLINTNEXTLINE(*-avoid-c-arrays)
constexpr AffinePoint G_ODD[size_t{1} << (G_WNAF_W - 2)] = {
#include "secp256k1_g_odd_w12.inc"
};
// NOLINTNEXTLINE(*-avoid-c-arrays)
constexpr AffinePoint PHI_G_ODD[size_t{1} << (G_WNAF_W - 2)] = {
#include "secp256k1_phig_odd_w12.inc"
};
/// Digits of a width-W NAF of a scalar below 2^128: the carry may add one more.
constexpr unsigned WNAF_LEN = 129;

/// Writes the width-W NAF of the scalar below 2^128 with 32-bit words w[0..3]: naf[i] is the
/// digit of 2^i, 0 or odd with |naf[i]| < 2^(W-1), and at least W-1 zeros follow each non-zero
/// one, so a 128-bit scalar has 128/(W+1) non-zero digits on average where its plain NAF has
/// 128/3. naf must be zeroed (WNAF_LEN digits). Returns the index past the top non-zero digit.
/// This is libsecp256k1's secp256k1_ecmult_wnaf().
template <unsigned W, typename Digit>
unsigned wnaf(Digit* naf, const uint32_t* w) noexcept
{
    const uint32_t x[6] = {w[0], w[1], w[2], w[3], 0, 0};
    // Trailing zeros of a non-zero word: rv32im has no ctz, so a de Bruijn lookup (5 instructions).
    static constexpr uint8_t DEBRUIJN[32] = {0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17,
        4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};
    const auto ctz = [](uint32_t v) noexcept {
        return static_cast<unsigned>(DEBRUIJN[((v & (0u - v)) * 0x077CB531u) >> 27]);
    };
    // The `count` bits at `pos`, from at most two adjacent words.
    const auto get_bits = [&x](unsigned pos, unsigned count) noexcept {
        const unsigned wi = pos / 32;
        const unsigned sh = pos % 32;
        uint32_t v = x[wi] >> sh;
        if (sh != 0)
            v |= x[wi + 1] << (32 - sh);
        return v & ((uint32_t{1} << count) - 1);
    };
    unsigned bit = 0;
    unsigned len = 0;
    uint32_t carry = 0;
    while (bit < WNAF_LEN)
    {
        // The next bit that differs from the carry, found a word at a time rather than bit by
        // bit: the bits equal to the carry are cleared and the lowest remaining one located.
        const unsigned wi = bit / 32;
        const uint32_t differing = (x[wi] ^ (0u - carry)) >> (bit % 32);
        if (differing == 0)
        {
            bit = (wi + 1) * 32;
            continue;
        }
        bit += ctz(differing);
        if (bit >= WNAF_LEN)
            break;
        const unsigned now = std::min(W, WNAF_LEN - bit);
        const auto word = get_bits(bit, now) + carry;
        carry = (word >> (W - 1)) & 1;
        naf[bit] = static_cast<Digit>(static_cast<int>(word) - static_cast<int>(carry << W));
        len = bit + 1;
        bit += now;
    }
    return len;
}

/// 1 in Montgomery form, folded at compile time (Fp::one() at run time is a CSR multiplication).
constexpr auto FP_ONE = Curve::Fp::one();

/// p = 2p in place: ecc::dbl()'s a = 0 formula written into p's own coordinates as each one
/// dies (x after S, z right away as Z' = 2YZ), skipping the copies into the returned point and
/// back (3 CSR MEMCOPY each way).
__attribute__((flatten)) void dbl_inplace(ecc::ProjPoint<Curve>& p) noexcept
{
    using FE = Curve::Fp;
    auto& [x1, y1, z1] = p;
    DECL_FE_COPY(FE, xx, x1); xx *= x1;          // X^2
    DECL_FE_COPY(FE, yy, y1); yy *= y1;          // Y^2
    z1 *= y1; z1 += z1;                          // Z' = 2YZ
    DECL_FE_COPY(FE, yyyy, yy); yyyy *= yy;      // Y^4
    yy *= x1; yy += yy; yy += yy;                // S = 4XY^2
    DECL_FE_COPY(FE, m, xx); m += xx; m += xx;   // M = 3X^2
    x1 = m; x1 *= m;                             // M^2
    x1 -= yy; x1 -= yy;                          // X' = M^2 - 2S
    yy -= x1;                                    // S - X'
    yyyy += yyyy; yyyy += yyyy;                  // 4Y^4
    y1 = m; y1 *= yy;                            // M(S - X')
    y1 -= yyyy; y1 -= yyyy;                      // Y' = M(S - X') - 8Y^4
}

/// p += (x2, y2), an affine point other than infinity, in place: ecc::add()'s mixed formula
/// written into p's own coordinates as each one dies, skipping the copies through the returned
/// point. Taking the coordinates apart lets a negated table point pass only its new y. With Live
/// the caller knows p is not infinity either, which saves the 8-word test of z. Returns true if
/// the sum is the point at infinity (p == -(x2, y2)), and then leaves p with z == 0.
template <bool Live = false>
__attribute__((flatten)) bool madd_inplace(
    ecc::ProjPoint<Curve>& p, const Curve::Fp& x2, const Curve::Fp& y2) noexcept
{
    using FE = Curve::Fp;
    auto& [x1, y1, z1] = p;
    if constexpr (!Live)
    {
        if (p == 0)
        {
            x1 = x2;
            y1 = y2;
            z1 = FP_ONE;
            return false;
        }
    }
    DECL_FE_COPY(FE, z1z1, z1); z1z1 *= z1;      // z1^2
    DECL_FE_COPY(FE, h, x2); h *= z1z1;          // u2 = x2 z1^2
    z1z1 *= z1; z1z1 *= y2;                      // s2 = y2 z1^3
    h -= x1;                                     // h = u2 - x1
    DECL_FE_COPY(FE, t1, h); t1 += h;            // 2h
    DECL_FE_COPY(FE, i, t1); i *= t1;            // i = (2h)^2
    z1z1 -= y1;                                  // t2 = s2 - y1
    if (h == 0) [[unlikely]]
    {
        if (z1z1 == 0)  // p == (x2, y2)
        {
            dbl_inplace(p);
            return false;
        }
        z1 = FE{};  // p == -(x2, y2): the sum is the point at infinity.
        return true;
    }
    DECL_FE_COPY(FE, r, z1z1); r += z1z1;        // r = 2 t2
    DECL_FE_COPY(FE, v, x1); v *= i;             // v = x1 i
    i *= h;                                      // j = h i
    x1 = r; x1 *= r;                             // r^2
    x1 -= i; x1 -= v; x1 -= v;                   // x3 = r^2 - j - 2v
    v -= x1;                                     // v - x3
    i *= y1;                                     // y1 j
    y1 = r; y1 *= v;                             // r (v - x3)
    y1 -= i; y1 -= i;                            // y3 = r (v - x3) - 2 y1 j
    z1 *= h; z1 += z1;                           // z3 = 2 z1 h
    return false;
}

/// p += (x2 : y2 : z2), a Jacobian point other than infinity, in place: ecc::add()'s
/// add-1998-cmo-2 formula written into p's own coordinates as each one dies (see madd_inplace()).
/// Returns true if the sum is the point at infinity, and then leaves p with z == 0.
template <bool Live = false>
__attribute__((flatten)) bool jadd_inplace(ecc::ProjPoint<Curve>& p, const Curve::Fp& x2,
    const Curve::Fp& y2, const Curve::Fp& z2) noexcept
{
    using FE = Curve::Fp;
    auto& [x1, y1, z1] = p;
    if constexpr (!Live)
    {
        if (p == 0)
        {
            x1 = x2;
            y1 = y2;
            z1 = z2;
            return false;
        }
    }
    DECL_FE_COPY(FE, z1z1, z1); z1z1 *= z1;      // z1^2
    DECL_FE_COPY(FE, z2z2, z2); z2z2 *= z2;      // z2^2
    DECL_FE_COPY(FE, u1, x1); u1 *= z2z2;        // u1 = x1 z2^2
    DECL_FE_COPY(FE, h, x2); h *= z1z1;          // u2 = x2 z1^2
    z1z1 *= z1; z1z1 *= y2;                      // s2 = y2 z1^3
    z2z2 *= z2; z2z2 *= y1;                      // s1 = y1 z2^3
    h -= u1;                                     // h = u2 - u1
    z1z1 -= z2z2;                                // r = s2 - s1
    if (h == 0) [[unlikely]]
    {
        if (z1z1 == 0)  // p == (x2 : y2 : z2)
        {
            dbl_inplace(p);
            return false;
        }
        z1 = FE{};  // p == -(x2 : y2 : z2): the sum is the point at infinity.
        return true;
    }
    DECL_FE_COPY(FE, hh, h); hh *= h;            // h^2
    u1 *= hh;                                    // v = u1 h^2
    hh *= h;                                     // h^3
    x1 = z1z1; x1 *= z1z1;                       // r^2
    x1 -= hh; x1 -= u1; x1 -= u1;                // x3 = r^2 - h^3 - 2v
    u1 -= x1;                                    // v - x3
    hh *= z2z2;                                  // s1 h^3
    y1 = z1z1; y1 *= u1;                         // r (v - x3)
    y1 -= hh;                                    // y3 = r (v - x3) - s1 h^3
    z1 *= z2; z1 *= h;                           // z3 = z1 z2 h
    return false;
}

/// u1*G + u2*R with u1 = k1a + k1b*lambda and u2 = k2a + k2b*lambda (signed halves), from the
/// NAFs of the halves: width-12 ones of k1a and k1b (naf_ga, naf_gb) over G_ODD and PHI_G_ODD,
/// negated by the sign of the half, and width-5 ones of k2a and k2b over ta and tb: ta[0..7] and
/// ta[8..15] hold (2j+1)*P_a and its negation, P_a = +/-R by the sign of k2a; tb likewise for
/// P_b = +/-phi(R). naf_len is past the top non-zero digit of all four.
ecc::ProjPoint<Curve> msm_wnaf(bool neg_ga, bool neg_gb, const int16_t* naf_ga,
    const int16_t* naf_gb, const int8_t* naf_a, const int8_t* naf_b, unsigned naf_len,
    const AffinePoint* ta, const AffinePoint* tb) noexcept
{
    const auto top = naf_len;

    ecc::ProjPoint<Curve> result;  // The point at infinity.
    bool started = false;          // Doubling the point at infinity is a wasted doubling.
    // The first addition sets the accumulator; the others know it is a live point, until one
    // cancels it: a partial sum P meeting -P, which crafted inputs can reach (an ECRECOVER call
    // takes any hash). The next addition then sets the accumulator again.
    const auto add = [&](const Curve::Fp& x, const Curve::Fp& y) noexcept {
        if (started)
            started = !madd_inplace<true>(result, x, y);
        else
        {
            result.x = x;
            result.y = y;
            result.z = FP_ONE;
            started = true;
        }
    };
    for (auto i = top; i-- != 0;)
    {
        if (started)
            dbl_inplace(result);

        if (const int d = naf_a[i]; d != 0)
        {
            const auto& pt = d > 0 ? ta[d >> 1] : ta[R_TABLE_SIZE + ((-d) >> 1)];
            add(pt.x, pt.y);
        }
        if (const int d = naf_b[i]; d != 0)
        {
            const auto& pt = d > 0 ? tb[d >> 1] : tb[R_TABLE_SIZE + ((-d) >> 1)];
            add(pt.x, pt.y);
        }

        if (const int d = naf_ga[i]; d != 0)
        {
            const auto& pt = G_ODD[(d > 0 ? d : -d) >> 1];
            if ((d < 0) != neg_ga)
                add(pt.x, -pt.y);
            else
                add(pt.x, pt.y);
        }
        if (const int d = naf_gb[i]; d != 0)
        {
            const auto& pt = PHI_G_ODD[(d > 0 ? d : -d) >> 1];
            if ((d < 0) != neg_gb)
                add(pt.x, -pt.y);
            else
                add(pt.x, pt.y);
        }
    }
    return result;
}

/// u1*G + u2*R for a single signature (the ECRECOVER precompile), which has no batch to share
/// inversions with: msm_wnaf()'s digits, with the odd multiples of R kept in Jacobian coordinates
/// (2R by doubling, 3R by a mixed addition, the rest by Jacobian ones) and phi applied as
/// (BETA X : Y : Z). No table addition can hit P == +/-Q (see ecrecover_batch()).
ecc::ProjPoint<Curve> ecrecover_msm_single(
    const uint256& u1, const uint256& u2, const AffinePoint& R) noexcept
{
    using Point = ecc::ProjPoint<Curve>;
    const auto [a1, b1] = ecc::decompose<Curve>(u1);
    const auto [a2, b2] = ecc::decompose<Curve>(u2);

    alignas(32) std::byte t_raw[R_TABLE_SIZE * sizeof(Point)];
    alignas(32) std::byte e_raw[2 * R_TABLE_SIZE * sizeof(Curve::Fp)];
    auto* const t = reinterpret_cast<Point*>(t_raw);
    auto* const bx = reinterpret_cast<Curve::Fp*>(e_raw);  // BETA X_j
    auto* const ny = bx + R_TABLE_SIZE;                      // -Y_j
    auto& r1 = *new (&t[0]) Point{};
    r1.x = R.x;
    r1.y = R.y;
    r1.z = FP_ONE;
    Point two_r = r1;
    dbl_inplace(two_r);
    madd_inplace<true>(*new (&t[1]) Point{two_r}, R.x, R.y);
    for (size_t j = 2; j < R_TABLE_SIZE; ++j)
        jadd_inplace<true>(*new (&t[j]) Point{t[j - 1]}, two_r.x, two_r.y, two_r.z);
    const auto beta = Curve::Fp{Curve::BETA};
    for (size_t j = 0; j < R_TABLE_SIZE; ++j)
    {
        new (&bx[j]) Curve::Fp{t[j].x};
        bx[j] *= beta;
        new (&ny[j]) Curve::Fp{-t[j].y};
    }

    alignas(4) int8_t naf_a[WNAF_LEN + 3]{};
    alignas(4) int8_t naf_b[WNAF_LEN + 3]{};
    alignas(4) int16_t naf_ga[WNAF_LEN + 1]{};
    alignas(4) int16_t naf_gb[WNAF_LEN + 1]{};
    const auto top = std::max(
        std::max(wnaf<R_WNAF_W>(naf_a, reinterpret_cast<const uint32_t*>(&a2.value)),
            wnaf<R_WNAF_W>(naf_b, reinterpret_cast<const uint32_t*>(&b2.value))),
        std::max(wnaf<G_WNAF_W>(naf_ga, reinterpret_cast<const uint32_t*>(&a1.value)),
            wnaf<G_WNAF_W>(naf_gb, reinterpret_cast<const uint32_t*>(&b1.value))));

    Point result;  // The point at infinity.
    bool started = false;  // As in msm_wnaf(): an addition that cancels the sum restarts it.
    const auto add = [&](const Curve::Fp& x, const Curve::Fp& y) noexcept {
        if (started)
            started = !madd_inplace<true>(result, x, y);
        else
        {
            result.x = x;
            result.y = y;
            result.z = FP_ONE;
            started = true;
        }
    };
    const auto add_jac = [&](const Curve::Fp& x, const Curve::Fp& y, const Curve::Fp& z) noexcept {
        if (started)
            started = !jadd_inplace<true>(result, x, y, z);
        else
        {
            result.x = x;
            result.y = y;
            result.z = z;
            started = true;
        }
    };
    for (auto i = top; i-- != 0;)
    {
        if (started)
            dbl_inplace(result);

        if (const int d = naf_a[i]; d != 0)
        {
            const auto j = static_cast<size_t>((d > 0 ? d : -d) >> 1);
            add_jac(t[j].x, (d < 0) != a2.sign ? ny[j] : t[j].y, t[j].z);
        }
        if (const int d = naf_b[i]; d != 0)
        {
            const auto j = static_cast<size_t>((d > 0 ? d : -d) >> 1);
            add_jac(bx[j], (d < 0) != b2.sign ? ny[j] : t[j].y, t[j].z);
        }
        if (const int d = naf_ga[i]; d != 0)
        {
            const auto& pt = G_ODD[(d > 0 ? d : -d) >> 1];
            if ((d < 0) != a1.sign)
                add(pt.x, -pt.y);
            else
                add(pt.x, pt.y);
        }
        if (const int d = naf_gb[i]; d != 0)
        {
            const auto& pt = PHI_G_ODD[(d > 0 ? d : -d) >> 1];
            if ((d < 0) != b1.sign)
                add(pt.x, -pt.y);
            else
                add(pt.x, pt.y);
        }
    }
    return result;
}
}  // namespace
#endif

void ecrecover_batch(std::span<const EcrecoverInput> in, std::span<std::optional<evmc::address>> out,
    RecoveryMode mode) noexcept
{
#if defined(AIRBENDER) && defined(__riscv)
    // The steps of secp256k1_ecdsa_recover() per signature, with its two inversions batched.
    using Fr = Curve::Fr;
    using Fp = Curve::Fp;
    const size_t n = in.size();
    AlignedArray<Fr> r_inv(n), fr_prefix(n), s(n), z(n);
    AlignedArray<Fp> rx(n), ry(n), qz(n), fp_prefix(n);
    AlignedArray<ecc::ProjPoint<Curve>> q(n);
    const std::unique_ptr<uint8_t[]> live{new uint8_t[n]};

    for (size_t i = 0; i < n; ++i)
    {
        out[i] = std::nullopt;
        live[i] = 0;
        const auto opt_r = Fr::from_bytes(in[i].r);
        if (!opt_r.has_value() || *opt_r == 0) [[unlikely]]
            continue;
        const auto opt_s = mode == RecoveryMode::strict ? Fr::from_bytes<Fr::Range::half>(in[i].s) :
                                                          Fr::from_bytes<Fr::Range::full>(in[i].s);
        if (!opt_s.has_value() || *opt_s == 0) [[unlikely]]
            continue;
        const auto r_mont = Fp{opt_r->value()};
        const auto y = calculate_y(r_mont, in[i].parity);
        if (!y.has_value()) [[unlikely]]
            continue;
        r_inv[i] = *opt_r;
        s[i] = *opt_s;
        z[i] = Fr{intx::be::unsafe::load<uint256>(in[i].hash.data())};
        rx[i] = r_mont;
        ry[i] = *y;
        live[i] = 1;
    }

    batch_invert(r_inv, fr_prefix, live.get(), n);  // r is in [1, n) and n is prime.

    // Split u1 and u2 by the endomorphism, and start on 2R in affine: lambda = 3x^2 / 2y needs
    // 1/(2y), batched (y != 0: secp256k1 has no point of order 2).
    AlignedArray<uint256> k1a(n), k1b(n), k2a(n), k2b(n);
    const std::unique_ptr<uint8_t[]> signs{new uint8_t[n]};
    AlignedArray<Fp> t(n);
    for (size_t i = 0; i < n; ++i)
    {
        if (!live[i])
            continue;
        const auto u1 = -z[i] * r_inv[i];
        const auto u2 = s[i] * r_inv[i];
        const auto [a1, b1] = ecc::decompose<Curve>(u1.value());
        const auto [a2, b2] = ecc::decompose<Curve>(u2.value());
        k1a[i] = a1.value;
        k1b[i] = b1.value;
        k2a[i] = a2.value;
        k2b[i] = b2.value;
        signs[i] = static_cast<uint8_t>(a1.sign | b1.sign << 1 | a2.sign << 2 | b2.sign << 3);
        t[i] = ry[i];
        t[i] += ry[i];
    }
    batch_invert(t, fp_prefix, live.get(), n);

    // The odd multiples 3R..15R: 2R in affine, then 7 mixed additions in Jacobian coordinates,
    // their z batched into one inversion for the whole block. No addition can hit P == +/-Q:
    // that needs (j +/- 2)R = 0 for some j <= 13, below the (prime) group order.
    constexpr size_t M = R_TABLE_SIZE - 1;
    AlignedArray<AffinePoint> two_r(n);
    AlignedArray<ecc::ProjPoint<Curve>> odd(n * M);
    AlignedArray<Fp> odd_z(n * M), odd_prefix(n * M);
    const std::unique_ptr<uint8_t[]> odd_live{new uint8_t[n * M]};
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < M; ++j)
            odd_live[i * M + j] = live[i];
        if (!live[i])
            continue;
        auto lambda = rx[i];
        lambda *= rx[i];
        const auto xx = lambda;
        lambda += xx;
        lambda += xx;
        lambda *= t[i];  // 3x^2 / 2y
        auto x2 = lambda;
        x2 *= lambda;
        x2 -= rx[i];
        x2 -= rx[i];  // lambda^2 - 2x
        auto y2 = rx[i];
        y2 -= x2;
        y2 *= lambda;
        y2 -= ry[i];  // lambda (x - x2) - y
        two_r[i] = AffinePoint{x2, y2};

        auto& p = odd[i * M];
        p.x = rx[i];
        p.y = ry[i];
        p.z = FP_ONE;
        for (size_t j = 0;; ++j)
        {
            auto& p_j = odd[i * M + j];
            madd_inplace<true>(p_j, two_r[i].x, two_r[i].y);
            odd_z[i * M + j] = p_j.z;
            if (j + 1 == M)
                break;
            odd[i * M + j + 1] = p_j;
        }
    }
    batch_invert(odd_z, odd_prefix, odd_live.get(), n * M);

    const auto beta = Fp{Curve::BETA};
    for (size_t i = 0; i < n; ++i)
    {
        if (!live[i])
            continue;
        // ta: (2j+1)*P_a then the negations, P_a = +/-R; tb likewise for P_b = +/-phi(R).
        alignas(32) std::byte ta_raw[2 * R_TABLE_SIZE * sizeof(AffinePoint)];
        alignas(32) std::byte tb_raw[2 * R_TABLE_SIZE * sizeof(AffinePoint)];
        auto* const ta = reinterpret_cast<AffinePoint*>(ta_raw);
        auto* const tb = reinterpret_cast<AffinePoint*>(tb_raw);
        const bool neg_a = (signs[i] >> 2) & 1;
        const bool neg_b = (signs[i] >> 3) & 1;
        const auto put = [&](size_t j, const Fp& x, const Fp& y) noexcept {
            const auto ny = -y;
            auto bx = x;
            bx *= beta;
            new (&ta[j]) AffinePoint{x, neg_a ? ny : y};
            new (&ta[R_TABLE_SIZE + j]) AffinePoint{x, neg_a ? y : ny};
            new (&tb[j]) AffinePoint{bx, neg_b ? ny : y};
            new (&tb[R_TABLE_SIZE + j]) AffinePoint{bx, neg_b ? y : ny};
        };
        put(0, rx[i], ry[i]);
        for (size_t j = 1; j < R_TABLE_SIZE; ++j)
        {
            // to_affine() with the batched z_inv.
            const auto& q_j = odd[i * M + j - 1];
            const auto& z_inv = odd_z[i * M + j - 1];
            auto zz_inv = z_inv;
            zz_inv *= z_inv;
            auto x = q_j.x;
            x *= zz_inv;
            zz_inv *= z_inv;
            auto y = q_j.y;
            y *= zz_inv;
            put(j, x, y);
        }

        alignas(4) int8_t naf_a[WNAF_LEN + 3]{};
        alignas(4) int8_t naf_b[WNAF_LEN + 3]{};
        alignas(4) int16_t naf_ga[WNAF_LEN + 1]{};
        alignas(4) int16_t naf_gb[WNAF_LEN + 1]{};
        const auto len_a = wnaf<R_WNAF_W>(naf_a, reinterpret_cast<const uint32_t*>(&k2a[i]));
        const auto len_b = wnaf<R_WNAF_W>(naf_b, reinterpret_cast<const uint32_t*>(&k2b[i]));
        const auto len_ga = wnaf<G_WNAF_W>(naf_ga, reinterpret_cast<const uint32_t*>(&k1a[i]));
        const auto len_gb = wnaf<G_WNAF_W>(naf_gb, reinterpret_cast<const uint32_t*>(&k1b[i]));

        q[i] = msm_wnaf(signs[i] & 1, (signs[i] >> 1) & 1, naf_ga, naf_gb, naf_a, naf_b,
            std::max(std::max(len_a, len_b), std::max(len_ga, len_gb)), ta, tb);
        if (q[i] == 0) [[unlikely]]  // The public key mustn't be the point at infinity.
        {
            live[i] = 0;
            continue;
        }
        qz[i] = q[i].z;
    }

    batch_invert(qz, fp_prefix, live.get(), n);  // Z != 0 for every remaining point.

    for (size_t i = 0; i < n; ++i)
    {
        if (!live[i])
            continue;
        // to_affine() with the batched z_inv.
        auto zz_inv = qz[i];
        zz_inv *= qz[i];
        auto zzz_inv = zz_inv;
        zzz_inv *= qz[i];
        auto x = q[i].x;
        x *= zz_inv;
        auto y = q[i].y;
        y *= zzz_inv;
        out[i] = to_address(AffinePoint{x, y});
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
#if defined(AIRBENDER) && defined(__riscv)
    // Uninit buffers avoid dead zero-init of 5 FieldElement vars (40 sw zero).
    DECL_UNINIT_BUF(Curve::Fp, z);
    DECL_UNINIT_BUF(Curve::Fp, t0);
    DECL_UNINIT_BUF(Curve::Fp, t1);
    DECL_UNINIT_BUF(Curve::Fp, t2);
    DECL_UNINIT_BUF(Curve::Fp, t3);
#else
    Curve::Fp z;
    Curve::Fp t0;
    Curve::Fp t1;
    Curve::Fp t2;
    Curve::Fp t3;
#endif


    // Step 1: z = x^0x2
    z = x; z *= x;                 // copy+mul_assign saves 1 MEMCOPY vs operator*

    // Step 2: z = x^0x3
    z *= x;

    // Step 4: t0 = x^0xc  (2 squarings of z)
#if defined(AIRBENDER) && defined(__riscv)
    // square_n_assign avoids wrap() overhead (~20 insns/call): no zero-init, no word copy, no return copy.
    t0 = z; t0.square_n_assign(2);
#else
    t0 = z.square_n(2);
#endif

    // Step 5: t0 = x^0xf
    t0 *= z;

    // Step 6: t1 = x^0x1e
    t1 = t0; t1 *= t0;            // copy+mul_assign saves 1 MEMCOPY vs operator*

    // Step 7: t2 = x^0x1f
    t2 = t1; t2 *= x;             // copy+mul_assign saves 1 MEMCOPY vs operator*

    // Step 9: t1 = x^0x7c  (2 squarings of t2)
#if defined(AIRBENDER) && defined(__riscv)
    t1 = t2; t1.square_n_assign(2);
#else
    t1 = t2.square_n(2);
#endif

    // Step 10: t1 = x^0x7f
    t1 *= z;

    // Step 14: t3 = x^0x7f0  (4 squarings of t1)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t1; t3.square_n_assign(4);
#else
    t3 = t1.square_n(4);
#endif

    // Step 15: t0 = x^0x7ff
    t0 *= t3;

    // Step 26: t3 = x^0x3ff800  (11 squarings of t0)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t0; t3.square_n_assign(11);
#else
    t3 = t0.square_n(11);
#endif

    // Step 27: t0 = x^0x3fffff
    t0 *= t3;

    // Step 32: t3 = x^0x7ffffe0  (5 squarings of t0)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t0; t3.square_n_assign(5);
#else
    t3 = t0.square_n(5);
#endif

    // Step 33: t2 = x^0x7ffffff
    t2 *= t3;

    // Step 60: t3 = x^0x3ffffff8000000  (27 squarings of t2)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t2; t3.square_n_assign(27);
#else
    t3 = t2.square_n(27);
#endif

    // Step 61: t2 = x^0x3fffffffffffff
    t2 *= t3;

    // Step 115: t3 = (54 squarings of t2)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t2; t3.square_n_assign(54);
#else
    t3 = t2.square_n(54);
#endif

    // Step 116: t2 = x^0xfffffffffffffffffffffffffff
    t2 *= t3;

    // Step 224: t3 = (108 squarings of t2)
#if defined(AIRBENDER) && defined(__riscv)
    t3 = t2; t3.square_n_assign(108);
#else
    t3 = t2.square_n(108);
#endif

    // Step 225: t2 = x^0xffffffffffffffffffffffffffffffffffffffffffffffffffffff
    t2 *= t3;

    // Step 232: t2 = (7 squarings)
#if defined(AIRBENDER) && defined(__riscv)
    t2.square_n_assign(7);
#else
    t2 = t2.square_n(7);
#endif

    // Step 233: t1 = x^0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffff
    t1 *= t2;

    // Step 256: t1 = (23 squarings)
#if defined(AIRBENDER) && defined(__riscv)
    t1.square_n_assign(23);
#else
    t1 = t1.square_n(23);
#endif

    // Step 257: t0 = x^0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff
    t0 *= t1;

    // Step 263: t0 = (6 squarings)
#if defined(AIRBENDER) && defined(__riscv)
    t0.square_n_assign(6);
#else
    t0 = t0.square_n(6);
#endif

    // Step 264: z = x^0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc3
    z *= t0;

    // Step 266: z = (2 squarings)
#if defined(AIRBENDER) && defined(__riscv)
    z.square_n_assign(2);
#else
    z = z.square_n(2);
#endif

    {
        auto zz = z; zz *= z;     // z^2 (copy+mul_assign saves 1 MEMCOPY vs operator*)
        if (zz != x)
            return std::nullopt;  // Computed value is not the square root.
    }

    return z;
}
}  // namespace evmone::crypto::secp256k1
