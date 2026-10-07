// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2023 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ecc.hpp"
#include "hash_types.h"
#include <evmc/evmc.hpp>
#include <optional>
#include <span>

namespace evmone::crypto::secp256k1
{
using namespace intx;

struct Curve
{
    using uint_type = uint256;

    struct FpSpec
    {
        /// The field prime number (P).
        static constexpr auto ORDER =
            0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f_u256;
    };
    using Fp = ecc::FieldElement<FpSpec>;

    struct FrSpec
    {
        /// The secp256k1 curve group order (N).
        static constexpr auto ORDER =
            0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141_u256;
    };
    using Fr = ecc::FieldElement<FrSpec>;

    static constexpr auto& FIELD_PRIME = Fp::ORDER;
    static constexpr auto& ORDER = Fr::ORDER;

    static constexpr auto A = 0;

    /// GLV endomorphism constants for secp256k1.
    /// The endomorphism phi(x,y) = (BETA*x, y) satisfies phi(P) = LAMBDA*P.

    /// Scalar eigenvalue of the endomorphism: phi(P) = LAMBDA * P.
    static constexpr auto LAMBDA =
        0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72_u256;

    /// Field element BETA such that phi(x,y) = (BETA*x, y).
    static constexpr auto BETA =
        0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee_u256;

    /// Lattice basis vectors for GLV scalar decomposition.
    /// v1 = (X1, -MINUS_Y1), v2 = (X2, Y2)
    static constexpr auto X1 =
        0x3086d221a7d46bcde86c90e49284eb15_u256;
    static constexpr auto MINUS_Y1 =
        0xe4437ed6010e88286f547fa90abfe4c3_u256;
    static constexpr auto X2 =
        0x114ca50f7a8e2f3f657c1108d9d44cfd8_u256;
    static constexpr auto Y2 =
        0x3086d221a7d46bcde86c90e49284eb15_u256;
};

using AffinePoint = ecc::AffinePoint<Curve>;

/// Square root for secp256k1 prime field.
///
/// Computes √x mod P by computing modular exponentiation x^((P+1)/4),
/// where P is ::FieldPrime.
///
/// @return Square root of x if it exists, std::nullopt otherwise.
std::optional<Curve::Fp> field_sqrt(const Curve::Fp& x) noexcept;

/// Calculate y coordinate of a point having x coordinate and y parity.
std::optional<Curve::Fp> calculate_y(const Curve::Fp& x, bool y_parity) noexcept;

/// Hash the secp256k1 uncompressed public key to Ethereum address.
evmc::address to_address(std::span<const uint8_t, 64> pubkey) noexcept;

/// Convert the secp256k1 point (uncompressed public key) to Ethereum address.
evmc::address to_address(const AffinePoint& pt) noexcept;

/// The strictness of the signer recovery from a signature.
enum class RecoveryMode : bool
{
    strict,     ///< Restrict s value range to lower half, prevents signature malleability (EIP-2).
    malleable,  ///< Full range for s value, signature is malleable.
};

std::optional<AffinePoint> secp256k1_ecdsa_recover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes, bool parity,
    RecoveryMode mode) noexcept;

/// Recovers the address that signed the message @p hash.
///
/// TODO: Make strict mode the default.
std::optional<evmc::address> ecrecover(std::span<const uint8_t, 32> hash,
    std::span<const uint8_t, 32> r_bytes, std::span<const uint8_t, 32> s_bytes, bool parity,
    RecoveryMode mode = RecoveryMode::malleable) noexcept;

/// One signature of an ecrecover_batch() call.
struct EcrecoverInput
{
    std::span<const uint8_t, 32> hash;
    std::span<const uint8_t, 32> r;
    std::span<const uint8_t, 32> s;
    bool parity;
};

/// Sets out[i] to ecrecover() of in[i]. On AIRBENDER the field inversions each signature needs
/// (r^-1 mod n; and mod p 1/(2y) of the recovered point, one per round of its affine odd-multiple
/// table, and the projective Z^-1) are shared by the whole batch with Montgomery's trick: one
/// Fermat inversion mod n and nine mod p (1/(2y), seven table rounds, the final Z) per batch, plus
/// 3 multiplications per signature and inversion. Elsewhere this calls ecrecover() for each
/// signature.
void ecrecover_batch(std::span<const EcrecoverInput> in, std::span<std::optional<evmc::address>> out,
    RecoveryMode mode = RecoveryMode::malleable) noexcept;

}  // namespace evmone::crypto::secp256k1
