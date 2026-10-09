// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2024 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "kzg.hpp"
#include "kzg_precomputed_lines.hpp"
#include <blst.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace evmone::crypto
{
namespace
{
/// The negation of the subgroup G1 generator -[1]₁ (affine coordinates in Montgomery form).
constexpr blst_p1_affine G1_GENERATOR_NEGATIVE{
    {0x5cb38790fd530c16, 0x7817fc679976fff5, 0x154f95c7143ba1c1, 0xf0ae6acdf3d0e747,
        0xedce6ecc21dbf440, 0x120177419e0bfb75},
    {0xff526c2af318883a, 0x92899ce4383b0270, 0x89d7738d9fa9d055, 0x12caf35ba344c12a,
        0x3cff1b76964b5317, 0x0e44d2ede9774430}};

/// Load and validate an element from the group order field.
std::optional<blst_scalar> validate_scalar(std::span<const std::byte, 32> b) noexcept
{
    blst_scalar v;
    blst_scalar_from_bendian(&v, reinterpret_cast<const uint8_t*>(b.data()));
    return blst_scalar_fr_check(&v) ? std::optional{v} : std::nullopt;
}

/// Uncompress and validate a point from G1 subgroup.
std::optional<blst_p1_affine> validate_G1(std::span<const std::byte, 48> b) noexcept
{
    blst_p1_affine r;
    if (blst_p1_uncompress(&r, reinterpret_cast<const uint8_t*>(b.data())) != BLST_SUCCESS)
        return std::nullopt;

    // Subgroup check is required by the spec but there are no test vectors
    // with points outside G1 which would satisfy the final pairings check.
    // The point at infinity is in G1 (the spec's validate_kzg_g1() accepts it
    // before the KeyValidate subgroup check), so skip the costly check for it.
    if (!blst_p1_affine_is_inf(&r) && !blst_p1_affine_in_g1(&r))
        return std::nullopt;
    return r;
}

/// Add two points from E1 and convert the result to affine form.
/// The conversion to affine is very costly so use only if the affine of the result is needed.
blst_p1_affine add_or_double(const blst_p1_affine& p, const blst_p1& q) noexcept
{
    blst_p1 r;
    blst_p1_add_or_double_affine(&r, &q, &p);
    blst_p1_affine ra;
    blst_p1_to_affine(&ra, &r);
    return ra;
}

/// Evaluates a precomputed Miller loop line at the G1 point P given as Px2 = (-2·P.x, 2·P.y),
/// like blst's post_line_by_Px2(). The result is a sparse Fp12 element in blst's "xy00z0" form.
void eval_line(blst_fp6& out, const blst_fp6& line, const blst_p1_affine& Px2) noexcept
{
    out.fp2[0] = line.fp2[0];
    blst_fp_mul(&out.fp2[1].fp[0], &line.fp2[1].fp[0], &Px2.x);
    blst_fp_mul(&out.fp2[1].fp[1], &line.fp2[1].fp[1], &Px2.x);
    blst_fp_mul(&out.fp2[2].fp[0], &line.fp2[2].fp[0], &Px2.y);
    blst_fp_mul(&out.fp2[2].fp[1], &line.fp2[2].fp[1], &Px2.y);
}

/// Checks e(a1, [1]₂) == e(b1, [s]₂) with a single Miller loop over both precomputed line tables.
///
/// blst_fp12_finalverify(GT1, GT2) exponentiates conj(GT1)·GT2 where each GT is the Miller loop
/// product F(P) conjugated, i.e. it exponentiates F(a1)·conj(F(b1)). Negating P.y negates only
/// the w-coefficient of every line, so F(-b1) = conj(F(b1)) and the product is F(a1)·F(-b1):
/// one loop that multiplies both lines into the same accumulator shares all 62 squarings.
bool pairings_verify(const blst_p1_affine& a1, const blst_p1_affine& b1) noexcept
{
    const blst_fp6* const a_lines = g2_gen_lines();          // [1]₂
    const blst_fp6* const b_lines = kzg_setup_g2_1_lines();  // [s]₂

    blst_p1_affine a_Px2;
    blst_fp_add(&a_Px2.x, &a1.x, &a1.x);
    blst_fp_cneg(&a_Px2.x, &a_Px2.x, true);
    blst_fp_add(&a_Px2.y, &a1.y, &a1.y);
    blst_p1_affine b_Px2;  // -b1
    blst_fp_add(&b_Px2.x, &b1.x, &b1.x);
    blst_fp_cneg(&b_Px2.x, &b_Px2.x, true);
    blst_fp_add(&b_Px2.y, &b1.y, &b1.y);
    blst_fp_cneg(&b_Px2.y, &b_Px2.y, true);

    // The first step is f = 1²·line = line.
    blst_fp6 line;
    eval_line(line, a_lines[0], a_Px2);
    blst_fp12 f{};
    f.fp6[0].fp2[0] = line.fp2[0];
    f.fp6[0].fp2[1] = line.fp2[1];
    f.fp6[1].fp2[1] = line.fp2[2];
    eval_line(line, b_lines[0], b_Px2);
    blst_fp12_mul_by_xy00z0(&f, &f, &line);

    const auto mul_lines = [&](size_t i) noexcept {
        eval_line(line, a_lines[i], a_Px2);
        blst_fp12_mul_by_xy00z0(&f, &f, &line);
        eval_line(line, b_lines[i], b_Px2);
        blst_fp12_mul_by_xy00z0(&f, &f, &line);
    };
    // The remaining 67 lines in the blocks of blst's miller_loop_lines(): the line at the block
    // start, then n times a squaring and the next line.
    static constexpr std::pair<uint8_t, uint8_t> BLOCKS[]{{1, 2}, {4, 3}, {8, 9}, {18, 32}, {51, 16}};
    // The blocks must take lines 1..67 in order (62 squarings), as both tables hold 68 lines.
    static_assert([] {
        size_t next = 1;
        for (const auto& [start, n] : BLOCKS)
        {
            if (start != next)
                return false;
            next = size_t{start} + n + 1;
        }
        return next == 68;
    }());
    for (const auto& [start, n] : BLOCKS)
    {
        mul_lines(start);
        for (size_t i = start + 1; i <= size_t{start} + n; ++i)
        {
            blst_fp12_sqr(&f, &f);
            mul_lines(i);
        }
    }

    // No final conjugation: the product already has the form finalverify exponentiates.
    blst_final_exp(&f, &f);
    return blst_fp12_is_one(&f);
}
}  // namespace

// NOTE: The Rust guest (guest_hypercube/src/precompiles.rs) defined rust_point_evaluation()
// which used the bls12_381 Rust crate for KZG point evaluation, bypassing the blst C library.
// In the pure C++ guest (som/remove-rust), that Rust symbol is unavailable, so we fall through
// to the blst-based implementation below. This may cost more cycles but is functionally correct.

bool kzg_verify_proof(const std::byte versioned_hash[VERSIONED_HASH_SIZE], const std::byte z[32],
    const std::byte y[32], const std::byte commitment[48], const std::byte proof[48]) noexcept
{
    std::byte computed_versioned_hash[32];
    sha256(computed_versioned_hash, commitment, 48);
    computed_versioned_hash[0] = VERSIONED_HASH_VERSION_KZG;
    if (!std::ranges::equal(std::span{versioned_hash, 32}, computed_versioned_hash))
        return false;

    // Load and validate scalars z and y.
    // TODO(C++26): The span construction can be done as std::snap(z, std::c_<32>).
    const auto zz = validate_scalar(std::span<const std::byte, 32>{z, 32});
    if (!zz)
        return false;
    const auto yy = validate_scalar(std::span<const std::byte, 32>{y, 32});
    if (!yy)
        return false;

    // Uncompress and validate the points C (representing the polynomial commitment)
    // and Pi (representing the proof). They both are valid to be points at infinity
    // when they prove a commitment to a constant polynomial,
    // see https://hackmd.io/@kevaundray/kzg-is-zero-proof-sound
    const auto C = validate_G1(std::span<const std::byte, 48>{commitment, 48});
    if (!C)
        return false;
    const auto Pi = validate_G1(std::span<const std::byte, 48>{proof, 48});
    if (!Pi)
        return false;

    // C = π = O commits to and proves the zero polynomial (e.g. of an empty blob).
    // Then the verification equation e(C - [y]₁, [1]₂) =? e(π, [s - z]₂) below becomes
    // e(-[y]₁, [1]₂) =? 1, which holds iff y = 0 (y < r is validated), for any z.
    // Decide it without the MSM and the pairings.
    if (blst_p1_affine_is_inf(&*C) && blst_p1_affine_is_inf(&*Pi))
        return std::ranges::all_of(std::span{y, 32}, [](std::byte b) { return b == std::byte{0}; });

    // The standard KZG verification equation
    //     e(C - [y]₁, [1]₂) =? e(π, [s - z]₂)
    // is rearranged via bilinearity into
    //     e(C + [z]π - [y]₁, [1]₂) =? e(π, [s]₂)
    // which eliminates the G2 multiplication and uses the 2-point MSM for G1.

    // Compute [z]π + [y](-[1]₁).
    const blst_p1_affine* const points[]{&*Pi, &G1_GENERATOR_NEGATIVE};
    const byte* const scalars[]{zz->b, yy->b};
    // For 2 points this actually doesn't use the Pippenger, and we can skip the scratch allocation.
    blst_p1 z_pi_minus_y_g1;
    blst_p1s_mult_pippenger(&z_pi_minus_y_g1, points, 2, scalars, BLS_MODULUS_BITS, nullptr);

    // Compute C + ([z]π - [y]₁). The addends may be the same / opposite points.
    const auto lsh_g1 = add_or_double(*C, z_pi_minus_y_g1);

    // e(C + [z]π - [y]₁, [1]₂) =? e(π, [s]₂)
    return pairings_verify(lsh_g1, *Pi);
}
}  // namespace evmone::crypto
