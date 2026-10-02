// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2025 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#include "secp256r1.hpp"

#ifdef ZISK
#include <zisk_precompiles.hpp>
#include <algorithm>
#endif

namespace evmone::crypto::secp256r1
{
namespace
{
#ifdef ZISK
constexpr auto Gx = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296_u256;
constexpr auto Gy = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5_u256;
constexpr uint64_t zisk_G[8] = {Gx[0], Gx[1], Gx[2], Gx[3], Gy[0], Gy[1], Gy[2], Gy[3]};

/// u1×G + u2×Q; all-zero is infinity.
void msm(uint64_t r[8], const uint256& u1, const uint256& u2, const uint64_t q[8]) noexcept
{
    uint64_t h[8];
    std::copy_n(zisk_G, 8, h);
    zisk::secp256r1_add_checked(h, q);
    const uint64_t* const points[4] = {nullptr, zisk_G, q, h};

    std::fill_n(r, 8, 0);
    for (auto i = std::max(intx::bit_width(u1), intx::bit_width(u2)); i != 0; --i)
    {
        // Prime order: no y == 0 points.
        if (!is_zero(r))
            zisk::secp256r1_dbl(r);
        const auto idx = 2 * unsigned{intx::bit_test(u2, i - 1)} + unsigned{intx::bit_test(u1, i - 1)};
        if (idx != 0)
            zisk::secp256r1_add_checked(r, points[idx]);
    }
}

/// verify() past the range checks.
bool verify_zisk(const ethash::hash256& h, const uint256& r, const uint256& s, const uint256& qx,
    const uint256& qy) noexcept
{
    constexpr auto& P = Curve::FIELD_PRIME;
    constexpr auto& N = Curve::ORDER;

    if (qx == 0 && qy == 0)
        return false;
    const auto x2_plus_a = zisk::arith256_mod(qx, qx, Curve::A, P);
    if (zisk::mulmod256(qy, qy, P) != zisk::arith256_mod(x2_plus_a, qx, Curve::B, P))
        return false;

    const auto z = intx::be::load<uint256>(h.bytes);
    const auto s_inv = zisk::secp256r1_fn_inv(s);
    const auto u1 = zisk::mulmod256(z, s_inv, N);
    const auto u2 = zisk::mulmod256(r, s_inv, N);

    const uint64_t q[8] = {qx[0], qx[1], qx[2], qx[3], qy[0], qy[1], qy[2], qy[3]};
    uint64_t R[8];
    msm(R, u1, u2, q);

    // Infinity gives x1 == 0 != r.
    uint256 x1{R[0], R[1], R[2], R[3]};
    if (x1 >= N)
        x1 -= N;
    return x1 == r;
}
#else
bool is_on_curve(const AffinePoint& p) noexcept
{
    static constexpr AffinePoint::FE A{Curve::A};
    static constexpr AffinePoint::FE B{Curve::B};
    return p.y * p.y == p.x * p.x * p.x + A * p.x + B;
}
#endif
}  // namespace

bool verify(const ethash::hash256& h, const uint256& r, const uint256& s, const uint256& qx,
    const uint256& qy) noexcept
{
    // The implementation follows "Elliptic Curve Digital Signature Algorithm"
    // https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm#Signature_verification_algorithm
    // but EIP-7951 spec is also a good source:
    // https://eips.ethereum.org/EIPS/eip-7951#signature-verification-algorithm

    // 1. Validate r and s are within [1, n-1].
    if (r == 0 || r >= Curve::ORDER || s == 0 || s >= Curve::ORDER)
        return false;

    // Check that Q is not equal to the identity element O, and its coordinates are otherwise valid.
    if (qx >= Curve::FIELD_PRIME || qy >= Curve::FIELD_PRIME)
        return false;
#ifdef ZISK
    return verify_zisk(h, r, s, qx, qy);
#else
    const AffinePoint Q{AffinePoint::FE{qx}, AffinePoint::FE{qy}};
    if (Q == 0)
        return false;

    // Check that Q lies on the curve.
    if (!is_on_curve(Q))
        return false;

    const ModArith n{Curve::ORDER};

    // 3. Let z be the Lₙ leftmost bits of e = HASH(m).
    static_assert(Curve::ORDER > 1_u256 << 255);
    const auto z = intx::be::load<uint256>(h.bytes);

    // 4. Calculate u₁ = zs⁻¹ mod n and u₂ = rs⁻¹ mod n.
    const auto s_inv = n.inv(n.to_mont(s));
    const auto u1 = n.from_mont(n.mul(n.to_mont(z), s_inv));
    const auto u2 = n.from_mont(n.mul(n.to_mont(r), s_inv));

    // 5. Calculate the curve point R = (x₁, y₁) = u₁×G + u₂×Q.
    const auto R = ecc::to_affine(msm(u1, G, u2, Q));

    //    If R is at infinity, the signature is invalid.
    //    In this case x₁ is 0 and cannot be equal to r.
    // 6. The signature is valid if r ≡ x₁ (mod n).
    auto x1 = R.x.value();
    if (x1 >= Curve::ORDER)
        x1 -= Curve::ORDER;

    return x1 == r;
#endif
}
}  // namespace evmone::crypto::secp256r1
