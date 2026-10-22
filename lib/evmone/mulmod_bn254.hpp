// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// SPDX-License-Identifier: Apache-2.0
#pragma once

/// The Montgomery reduction of a 512-bit MULMOD product by the two alt_bn128 (BN254) primes, on
/// the Airbender BigInt CSR (0x7ca). Profiling of mainnet blocks (a mulmod_stats build,
/// 2026-10-22) shows nearly every generic Knuth reduction — 28.6% of block 24572234's cycles,
/// 16.8% of 24576071's — multiplies by one of these two constants: a Groth16-style verifier in a
/// contract reduces by the BN254 scalar-field prime Fr thousands of times per block (and by the
/// base-field prime Fq alongside it). With the modulus known at compile time the Montgomery
/// constants cost nothing at run time: no cache, no fill — two passes of word reduction against
/// hard-coded constants, R = 2^256:
///   pass 1 (two word iterations): a = REDC2(x*y) = x*y*R^-2 mod r
///   pass 2 (one word iteration):  REDC1(a*rr3) = a*R^3*R^-1 = x*y mod r, rr3 = R^3 mod r
/// Each word iteration is four BigInt calls (u = t*minv; u*r low and high; accumulator += the
/// high half); the carry that the CSR ADD's mod-R wrap loses is read from the words (t + u*r mod R
/// is R unless both words are 0). A single-iteration pass can leave the result just above R (the
/// accumulator plus the high half of u*r), so its final subtract folds the carry word back with
/// rm = R - r. Every other modulus goes to the generic path (div32's urem).
///
/// Every pointer handed to the CSR is a named alignas(32) local, a stack slot of the caller or a
/// member of the aligned constant structs below: the delegation rejects unaligned pointers.
///
/// The reduction was verified in Python against big-int modular multiplication (3,000 random cases
/// per prime); the constants were generated there.

#include <intx/intx.hpp>
#include <cstdint>

#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32

namespace evmone::mont
{
using intx::uint256;

/// One BigInt CSR operation: *mut op= *imm, per the op mask of 0x7ca, as in instructions.hpp.
[[gnu::always_inline]] inline void csr_op(uint32_t op, void* mut, const void* imm) noexcept
{
    register uintptr_t r10 asm("x10") = reinterpret_cast<uintptr_t>(mut);
    register uintptr_t r11 asm("x11") = reinterpret_cast<uintptr_t>(imm);
    register uint32_t r12 asm("x12") = op;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
}
[[gnu::always_inline]] inline void csr_add(uint256& mut, const uint256& imm) noexcept
{
    csr_op(0x01, &mut, &imm);
}
[[gnu::always_inline]] inline void csr_sub(uint256& mut, const uint256& imm) noexcept
{
    csr_op(0x02, &mut, &imm);
}
[[gnu::always_inline]] inline void csr_mul_low(uint256& mut, const uint256& imm) noexcept
{
    csr_op(0x08, &mut, &imm);
}
[[gnu::always_inline]] inline void csr_mul_high(uint256& mut, const uint256& imm) noexcept
{
    csr_op(0x10, &mut, &imm);
}

/// The constants of one prime: mod, minv = -mod^-1 mod R, rr3 = R^3 mod mod, rm = R - mod. All
/// 32-byte aligned: each of them reaches the CSR.
struct alignas(32) BN254Consts
{
    uint256 mod;
    uint256 minv;
    uint256 rr3;
    uint256 rm;
};

/// The alt_bn128 scalar-field prime Fr
/// 21888242871839275222246405745257275088548364400416034343698204180538180272055.
inline constexpr BN254Consts kFr{
    uint256{0x30644e72e131a029ULL, 0xb85045b68181585dULL, 0x2833e84879b97091ULL, 0x43e1f593f0000001ULL},
    uint256{0xc2e1f593efffffffULL, 0x6586864b4c6911b3ULL, 0xe39a982899062391ULL, 0x73f82f1d0d8341b2ULL},
    uint256{0x5e94d8e1b4bf0040ULL, 0x2a489cbe1cfbb6b8ULL, 0x893cc664a19fcfedULL, 0x0cf8594b7fcc657cULL},
    uint256{0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL}};

/// The alt_bn128 base-field prime Fq
/// 21888242871839275222246405745257275088696311157297823664884691489454159642837.
inline constexpr BN254Consts kFq{
    uint256{0x30644e72e131a029ULL, 0xb85045b68181585dULL, 0x97816a916871ca8dULL, 0x3c208c16d87cfd47ULL},
    uint256{0x87d20782e4866389ULL, 0x9ede7d651eca6ac9ULL, 0xd8afcbd01833da80ULL, 0xf57a22b791888c6bULL},
    uint256{0xb1cd6dafda1530dfULL, 0x62f210e6a7283db6ULL, 0xef7f0b0c0ada0afbULL, 0x20fd6e902d592544ULL},
    uint256{0xc3df73e9278302b9ULL, 0x687e956e978e3572ULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL}};

/// Whether m is one of the two primes the fast path takes.
[[gnu::always_inline]] inline bool is_bn254_prime(const uint256& m) noexcept
{
    return m == kFr.mod || m == kFq.mod;
}

/// m = x * y % m from the 512-bit product p into m's slot, for m = kFr.mod or kFq.mod (both odd).
/// Two reduction passes as at the top of the file; ~15 BigInt calls against the generic path's
/// ~1,400-cycle Knuth division. All of m is read (the m == prime test) before the store into it.
[[gnu::noinline]] inline void mulmod_bn254(const intx::uint512& p, uint256& m) noexcept
{
    const BN254Consts& c = m == kFr.mod ? kFr : kFq;

    // Pass 1: a = REDC2(p) = p*R^-2 mod r, two word iterations over (t0, t1, t2-carry).
    alignas(32) uint256 t0;
    std::memcpy(&t0, &p, sizeof(uint256));
    alignas(32) uint256 t1;
    std::memcpy(&t1, reinterpret_cast<const uint8_t*>(&p) + sizeof(uint256), sizeof(uint256));
    uint256 t2 = 0;  // The carry word above t1: at most 2 during the first iteration.
    for (int i = 0; i != 2; ++i)
    {
        alignas(32) uint256 u = t0;
        csr_mul_low(u, c.minv);  // u = t0 * minv mod R
        alignas(32) uint256 lo = u;
        csr_mul_low(lo, c.mod);
        alignas(32) uint256 hi = u;
        csr_mul_high(hi, c.mod);
        // The low word drops: t0 + lo is 0 or R — R unless both are 0.
        const bool c1 = t0 != 0 || lo != 0;
        // t1 += hi + c1, carries into t2 by hand: the CSR ADD wraps mod R.
        alignas(32) uint256 s = t1;
        csr_add(s, hi);
        unsigned carry = s < t1 ? 1u : 0u;
        if (c1)
        {
            const uint256 before = s;
            s += uint256{1};
            carry += s < before ? 1u : 0u;
        }
        t2 = carry;
        t0 = s;
        t1 = t2;  // Shift down a word; the second iteration reduces (t1, 0).
        t2 = 0;
    }
    // At most c.mod: subtract once.
    if (t0 >= c.mod)
        csr_sub(t0, c.mod);
    const uint256 a = t0;

    // Pass 2: REDC1(a*rr3) = x*y mod r, one word iteration, the result folded below R with rm.
    alignas(32) uint256 plo = a;
    csr_mul_low(plo, c.rr3);
    alignas(32) uint256 phi = a;
    csr_mul_high(phi, c.rr3);
    alignas(32) uint256 u = plo;
    csr_mul_low(u, c.minv);
    alignas(32) uint256 lo = u;
    csr_mul_low(lo, c.mod);
    alignas(32) uint256 hi = u;
    csr_mul_high(hi, c.mod);
    const bool c2 = plo != 0 || lo != 0;
    alignas(32) uint256 s = phi;
    csr_add(s, hi);
    unsigned carry = s < phi ? 1u : 0u;
    if (c2)
    {
        const uint256 before = s;
        s += uint256{1};
        carry += s < before ? 1u : 0u;
    }
    // T' = s + carry*R < 2*mod. The carry case subtracts mod by adding rm = R - mod (the wrapped
    // s plus R - mod is T' - mod, below mod); otherwise one conditional subtract.
    if (carry)
        csr_add(s, c.rm);
    else if (s >= c.mod)
        csr_sub(s, c.mod);
    m = s;
}
}  // namespace evmone::mont

#endif  // AIRBENDER rv32
