// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// SPDX-License-Identifier: Apache-2.0
#pragma once

/// The Montgomery reduction of a 512-bit MULMOD product by an odd 256-bit modulus, on the
/// Airbender BigInt CSR (0x7ca). intx::internal::div32::mulmod_reduce reduces the products of the
/// sparse moduli (the NIST P-256 prime, 2^256 - 1, powers of two) in closed form and everything
/// else by Knuth division of up to 16 words by 8. Real programs multiply by the same modulus over
/// and over (a field loop, a modexp, a hash-to-curve), so the modulus and its Montgomery constants
/// are cached per modulus and the reduction itself runs on the CSR: one pass of REDC is five BigInt
/// calls and a few 256-bit additions in C, against a Knuth division of about eight quotient words
/// with 32-bit reciprocal estimates and submuls. The cache is adaptive: the first sighting of a
/// modulus reduces by Knuth division, the second fills the slot's constants, and every sighting
/// after that runs on the CSR — a modulus the program never repeats costs one cache scan.
///
/// R = 2^256. A product x*y below R^2 reduces in two passes, without converting the operands:
///   REDC2(x*y)  = x*y*R^-2 mod m            (the product's low and high halves are in hand; two
///                                             word iterations divide by R^2)
///   REDC2(a*rr) = x*y*R^-2 * R^4 * R^-2     = x*y mod m, with rr = R^4 mod m cached
/// The constants of a new modulus cost 10 BigInt calls of Newton inversion for -m^-1 mod R, 512
/// conditional doublings for R^2 mod m and one 512-bit square under div32's urem for R^4: all
/// comparable to a few Knuth divisions, paid once per modulus.
///
/// Every pointer handed to the CSR must be 32-byte aligned and point to RAM, like the EVM stack
/// slots (StackSpace alignas) and the cache below in .bss: local uint256 operands are declared
/// alignas(32).
///
/// Z6M_MULMOD_MONT_TEST compiles this on the host with the CSR ops emulated in C++.

#include <intx/intx.hpp>
#include <array>
#include <cstdint>
#include <cstring>

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(Z6M_MULMOD_MONT_TEST)

namespace evmone::mont
{
using intx::uint256;

/// One BigInt CSR operation: *mut op= *imm (or the product's half), per the op mask of 0x7ca.
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
[[gnu::always_inline]] inline void csr_op(uint32_t op, void* mut, const void* imm) noexcept
{
    register uintptr_t r10 asm("x10") = reinterpret_cast<uintptr_t>(mut);
    register uintptr_t r11 asm("x11") = reinterpret_cast<uintptr_t>(imm);
    register uint32_t r12 asm("x12") = op;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
}
#else
inline void csr_op(uint32_t op, void* mut, const void* imm) noexcept
{
    const auto m = static_cast<uint256*>(mut);
    const auto i = static_cast<const uint256*>(imm);
    switch (op)
    {
    case 0x01:  // ADD
        *m += *i;
        break;
    case 0x02:  // SUB
        *m -= *i;
        break;
    default:
    {
        // MUL_LOW (0x08) / MUL_HIGH (0x10), by half of the 512-bit product.
        const auto pr = intx::umul(*m, *i);
        const auto off = op == 0x08 ? 0 : 4;
        for (size_t k = 0; k != 4; ++k)
            (*m)[k] = pr[off + k];
        break;
    }
    }
}
#endif

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

/// The inverse of an odd byte mod 2^8: the Newton seed of -m^-1 mod R.
inline constexpr std::array<uint8_t, 256> inv_byte = [] {
    std::array<uint8_t, 256> t{};
    for (int i = 1; i < 256; i += 2)
    {
        unsigned x = 1;
        for (int j = 0; j != 5; ++j)  // x(2 - a x) doubles the order: 8 -> 256 in 5 steps.
            x = x * (2 - i * x) & 0xFF;
        t[i] = static_cast<uint8_t>(x);
    }
    return t;
}();

/// A cached modulus and its Montgomery constants: m (0 marks a free slot), -m^-1 mod R and
/// rr = R^4 mod m (a pass of redc is a multiplication by R^-2, so the second pass multiplies by
/// R^4 to land back on R^0). The BigInt CSR requires 32-byte-aligned operands. hits counts the
/// sightings: the constants are only worth their fill cost (a few divisions) when the modulus
/// repeats, so the first sighting reduces by Knuth division and the second fills the slot.
struct alignas(32) Entry
{
    uint256 m{};
    uint256 minv{};
    uint256 rr{};
    uint32_t hits{0};
    uint32_t ready{0};
};

/// One cache for the whole guest run (single-threaded): contracts repeat their moduli within a
/// call frame and across frames. A most-recently-used shortcut, which a field loop hits without
/// scanning, then a linear scan over a round-robin victim.
inline Entry cache[4];
inline Entry* cache_mru = &cache[0];
inline unsigned cache_clock = 0;

inline Entry* find_cache(const uint256& m) noexcept
{
    if (cache_mru->m == m)
        return cache_mru;
    for (auto& e : cache)
    {
        if (e.m == m)
            return cache_mru = &e;
    }
    return nullptr;
}

/// Fills the constants of the odd modulus m in its slot e, on the modulus's second sighting:
/// minv by Newton iteration from the low byte's inverse (5 steps double the order 2^8 -> 2^256),
/// rr = R^4 mod m = ((R mod m)^2)^2 mod m — R mod m by one division of the sparse power, then two
/// 512-bit squares under div32's urem. A few thousand cycles, once per modulus.
[[gnu::noinline]] inline void fill_entry(Entry& e, const uint256& m) noexcept
{
    alignas(32) uint256 x = inv_byte[reinterpret_cast<const uint8_t*>(&m)[0]];
    for (int i = 0; i != 5; ++i)
    {
        alignas(32) uint256 t = m;
        csr_mul_low(t, x);  // t = m*x mod R
        t = 2 - t;
        csr_mul_low(x, t);  // x = x*t mod R
    }
    const uint256 minv = -x;  // -m^-1 mod R

    // R mod m: 2^256 is one set bit in word 8 of a 512-bit object; div32's urem takes it whole.
    alignas(32) uint256 r{};
    {
        intx::uint512 pow{intx::uint512::uninit_tag{}};
        auto* const zw = reinterpret_cast<intx::internal::div32::w32*>(&pow);
        for (size_t i = 0; i < 16; ++i)
            zw[i] = 0;
        reinterpret_cast<uint8_t*>(&pow)[sizeof(uint256)] = 1;  // Byte 256: the bit of 2^256.
        intx::internal::div32::urem(pow,
            intx::internal::div32::significant_words<16>(zw),
            m, r);
    }
    // rr = R^4 mod m = ((R mod m)^2)^2 mod m, by the two divisions the fill may spend.
    for (int i = 0; i != 2; ++i)
    {
        const auto sq = intx::umul(r, r);
        intx::internal::div32::urem(sq,
            intx::internal::div32::significant_words<16>(
                reinterpret_cast<const intx::internal::div32::w32*>(&sq)),
            m, r);
    }
    e.minv = minv;
    e.rr = r;
    e.ready = 1;
}

/// r = T*R^-2 mod e.m for T = t0 + t1*R < R^2 (both words below R) and odd e.m: two iterations of
/// word Montgomery reduction with R = 2^256 as the word. Iteration i computes u = t_i * minv, adds
/// u*m to T above t_i (the low word of the sum is 0 mod R by construction, so t_i drops and its
/// carry is whether the sum equals R), and shifts. Two iterations divide by R^2, which is why the
/// cached constant is R^4. The final word is at most e.m, so one conditional subtract ends it.
[[gnu::noinline]] inline uint256 redc(const uint256& t0_in, const uint256& t1_in, const Entry& e) noexcept
{
    alignas(32) uint256 t0 = t0_in;
    alignas(32) uint256 t1 = t1_in;
    uint256 t2 = 0;  // The carry word above t1: at most 1 after the first iteration.
    for (int i = 0; i != 2; ++i)
    {
        alignas(32) uint256 u = t0;
        csr_mul_low(u, e.minv);  // u = t0 * minv mod R
        alignas(32) uint256 lo = u;
        csr_mul_low(lo, e.m);
        alignas(32) uint256 hi = u;
        csr_mul_high(hi, e.m);
        // The low word drops: t0 + lo is 0 mod R by construction and both words are below R, so
        // the sum is 0 or R — R unless both are 0. The CSR ADD wraps mod R and could not tell
        // them apart; the words themselves do, with no call.
        const bool c = t0 != 0 || lo != 0;
        // t1 += hi + c, the overflow into t2 by hand: the CSR ADD wraps mod R. The 0/1 c goes
        // through plain arithmetic: GCC gave the constant's slot no 32-byte alignment, and the
        // BigInt CSR rejects unaligned pointers.
        const uint256 before = t1;
        csr_add(t1, hi);
        uint256 c1 = t1 < before ? uint256{1} : uint256{0};
        if (c)
        {
            const uint256 before1 = t1;
            t1 += uint256{1};
            if (t1 < before1)
                c1 += 1;
        }
        t2 += c1;
        // Shift down a word: the next iteration reduces (t1, t2).
        t0 = t1;
        t1 = t2;
        t2 = 0;
    }
    // At most e.m: subtract once.
    if (t0 >= e.m)
        csr_sub(t0, e.m);
    return t0;
}

/// m = x * y % m from the 512-bit product p, into m's slot, for an odd m the closed forms of
/// div32's mulmod_reduce_closed do not take: the cached Montgomery reduction (see the file
/// comment). Even moduli go to div32's urem. All of m is read before the store into it at the end.
[[gnu::noinline]] inline void mulmod_mont(const intx::uint512& p, uint256& m) noexcept
{
    const auto* const pw = reinterpret_cast<const uint8_t*>(&p);
    if ((reinterpret_cast<const uint8_t*>(&m)[0] & 1) == 0) [[unlikely]]
    {
        // Montgomery needs an odd modulus; div32's Knuth division takes any one.
        intx::internal::div32::urem(p,
            intx::internal::div32::significant_words<16>(
                reinterpret_cast<const intx::internal::div32::w32*>(pw)),
            m, m);
        return;
    }

    Entry* e = find_cache(m);
    if (e == nullptr)
    {
        // First sighting: remember the modulus, but reduce by Knuth division — the constants'
        // fill cost is only worth it when the modulus repeats.
        e = &cache[cache_clock++ & 3];
        e->m = m;
        e->hits = 1;
        e->ready = 0;
        cache_mru = e;
        intx::internal::div32::urem(p,
            intx::internal::div32::significant_words<16>(
                reinterpret_cast<const intx::internal::div32::w32*>(pw)),
            m, m);
        return;
    }
    cache_mru = e;
    if (!e->ready)
    {
        if (++e->hits < 2)
        {
            intx::internal::div32::urem(p,
                intx::internal::div32::significant_words<16>(
                    reinterpret_cast<const intx::internal::div32::w32*>(pw)),
                m, m);
            return;
        }
        fill_entry(*e, m);
    }

    // Pass 1: a = REDC(x*y) = x*y*R^-2 mod m.
    alignas(32) uint256 lo;
    std::memcpy(&lo, pw, sizeof(uint256));
    alignas(32) uint256 hi;
    std::memcpy(&hi, pw + sizeof(uint256), sizeof(uint256));
    const uint256 a = redc(lo, hi, *e);

    // Pass 2: REDC(a*rr) = x*y*R^-2 * R^4 * R^-2 = x*y mod m.
    alignas(32) uint256 plo = a;
    csr_mul_low(plo, e->rr);
    alignas(32) uint256 phi = a;
    csr_mul_high(phi, e->rr);
    m = redc(plo, phi, *e);
}
}  // namespace evmone::mont

#endif  // AIRBENDER rv32 or Z6M_MULMOD_MONT_TEST
