// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2023 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <intx/intx.hpp>
#include <cassert>

#ifdef SP1
#include <sp1_syscalls.hpp>
#endif

namespace evmmax
{
/// Compute the modular inverse of the number modulo 2³²: inv⋅a = 1 mod 2³².
constexpr uint32_t modinv(uint32_t a) noexcept
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

/// Compute the modular inverse of the number modulo 2⁶⁴: inv⋅a = 1 mod 2⁶⁴.
constexpr uint64_t modinv(uint64_t a) noexcept
{
    assert(a % 2 == 1);  // The argument must be odd, otherwise the inverse does not exist.
    uint64_t inv = modinv(static_cast<uint32_t>(a));  // Start with inversion mod 2³².
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
    return -modinv(mod[0]);
}

#if defined(AIRBENDER) && defined(__riscv)
/// Compute the full 256-bit Montgomery inverse: N' such that mod⋅N' ≡ -1 (mod 2²⁵⁶).
/// Uses Newton-Raphson starting from the 64-bit inverse, doubling bits each step.
template <typename UintT>
constexpr UintT compute_mont_mod_inv_full(const UintT& mod) noexcept
{
    // Start with 64-bit inverse
    UintT inv{};
    inv[0] = compute_mont_mod_inv(mod);
    // Newton-Raphson: for N' where N*N' ≡ -1 (mod R), step is N' * (2 + N*N')
    // Each iteration doubles the number of correct bits: 64 → 128 → 256
    inv = inv * (UintT{2} + mod * inv);  // 128 bits correct
    inv = inv * (UintT{2} + mod * inv);  // 256 bits correct
    return inv;
}
#endif

constexpr std::pair<uint64_t, uint64_t> addmul(
    uint64_t t, uint64_t a, uint64_t b, uint64_t c) noexcept
{
    const auto p = intx::umul(a, b) + t + c;
    return {p[1], p[0]};
}

/// The modular arithmetic operations for EVMMAX (EVM Modular Arithmetic Extensions).
template <typename UintT, bool BN = false>
class ModArith
{
#if defined(AIRBENDER) && defined(__riscv)
    // Align members to 32 bytes for direct use with BigInt CSR (avoids copies).
    alignas(32) const UintT mod_;  ///< The modulus.
    alignas(32) const UintT r_squared_;  ///< R² % mod.
    /// The modulus inversion, i.e. the number N' such that mod⋅N' = 2⁶⁴-1.
    const uint64_t mod_inv_;
    /// Full 256-bit Montgomery inverse: mod⋅mod_inv_full_ ≡ -1 (mod 2²⁵⁶).
    alignas(32) const UintT mod_inv_full_;
#else
    const UintT mod_;  ///< The modulus.
    const UintT r_squared_;  ///< R² % mod.
    /// The modulus inversion, i.e. the number N' such that mod⋅N' = 2⁶⁴-1.
    const uint64_t mod_inv_;
#endif

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
#elif defined(AIRBENDER) && defined(__riscv)
        r_squared_{compute_r_squared(mod)},
        mod_inv_{compute_mont_mod_inv(mod)},
        mod_inv_full_{compute_mont_mod_inv_full(mod)}
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

#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // Optimized from_mont: mul(x, 1) means T = x*1, so t_lo = x, t_hi = 0.
                // Skip the two MUL CSR calls for x*y entirely.
                // m = t_lo * N' mod 2^256 = x * N' mod 2^256
                // result = (0 + mN_hi + carry) where carry = (x != 0)
                // Then conditional subtract mod.

                alignas(32) UintT A{};     // t_hi = 0 -> result
                alignas(32) UintT D = x;   // m = x * N' (will be computed in place)

                // 1. m = MUL_LOW(x, N') -> D = m (use aligned mod_inv_full_ directly)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&D);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                    register uint32_t a2 asm("x12") = 0x08; // MUL_LOW
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 2. mN_hi = MUL_HIGH(m, N) -> D = mN_hi (use aligned mod_ directly)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&D);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x10; // MUL_HIGH
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 3. carry = (x != 0) since t_lo = x and t_lo + mN_lo = 0 or 2^256
                const uint32_t low_carry = (x != UintT{0}) ? 1u : 0u;

                // 4. result = 0 + mN_hi + carry -> A += D
                uint32_t carry;
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&D);
                    register uint32_t a2 asm("x12") = 0x01 | (low_carry << 6);
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    carry = a2;
                }

                // 5. Conditional subtract mod (use aligned mod_ directly)
                if (carry)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x02;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                else
                {
                    uint32_t borrow;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x02;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                        borrow = a2;
                    }
                    if (borrow)
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x01;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }
                }

                return A;
            }
        }
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

#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // BigInt CSR Montgomery multiplication using 3 aligned buffers (A, C, D):
                //   T = x*y  (512-bit)
                //   m = T_lo * N'  (mod 2^256)
                //   result = (T + m*N) >> 256
                //   if result >= N: result -= N
                //
                // Class members mod_, mod_inv_full_ are aligned, used directly as x11.
                // FieldElement value_ is aligned, so y can be used directly as x11.
                // Only 2 stack buffers needed: A (x10 mutable), B (scratch).

                alignas(32) UintT A;       // x -> t_lo -> t_hi -> result
                alignas(32) UintT B;       // scratch: t_lo saved -> m -> mn_hi

                // 0. Copy x -> A using MEMCOPY CSR
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
                    register uint32_t a2 asm("x12") = 0x80; // MEMCOPY
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 1. T_lo = MUL_LOW(x, y)  →  A = t_lo (y used directly as x11)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&y);
                    register uint32_t a2 asm("x12") = 0x08;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 2. Save t_lo: MEMCOPY A -> B, then compute m in B
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&A);
                    register uint32_t a2 asm("x12") = 0x80; // MEMCOPY A -> B
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 3. Reload x via MEMCOPY (1 cycle vs 16 instructions for word-by-word)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
                    register uint32_t a2 asm("x12") = 0x80; // MEMCOPY x -> A
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&y);
                    register uint32_t a2 asm("x12") = 0x10; // MUL_HIGH
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                // Now A = t_hi, B = t_lo

                // 4. Carry from low half (before overwriting B)
                const uint32_t low_carry = (B != UintT{0}) ? 1u : 0u;

                // 5. m = MUL_LOW(t_lo, N')  →  B = m
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                    register uint32_t a2 asm("x12") = 0x08; // MUL_LOW
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 6. mN_hi = MUL_HIGH(m, N)  →  B = mN_hi
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x10;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                // 7. result = ADD(t_hi + mN_hi + carry)  →  A += B
                uint32_t carry;
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&B);
                    register uint32_t a2 asm("x12") = 0x01 | (low_carry << 6);
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    carry = a2;
                }

                // 7. Conditional subtract: try SUB(A, mod), keep if no borrow
                //    Use aligned mod_ directly as x11.
                if (carry)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x02; // SUB
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                else
                {
                    uint32_t borrow;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x02;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                        borrow = a2;
                    }
                    if (borrow)
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x01;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }
                }

                return A;
            }
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

    /// Performs a modular addition. It is required that x < mod and y < mod, but x and y may be
    /// but are not required to be in Montgomery form.
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

#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // add = x + y, then try subtract mod. 2-3 CSR calls.
                // Uses aligned mod_ and y directly as x11.
                alignas(32) UintT res = x;
                uint32_t add_carry;
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&y);
                    register uint32_t a2 asm("x12") = 0x01; // ADD
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    add_carry = a2;
                }
                // Try subtract mod (use aligned mod_ directly)
                if (add_carry)
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x02; // SUB
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                else
                {
                    uint32_t borrow;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x02; // SUB
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                        borrow = a2;
                    }
                    if (borrow)
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x01; // ADD
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }
                }
                return res;
            }
        }
#endif

        const auto s = addc(x, y);  // TODO: cannot overflow if modulus is sparse (e.g. 255 bits).
        const auto d = subc(s.value, mod_);
        return (!s.carry && d.carry) ? s.value : d.value;
    }

    /// Performs a modular subtraction. It is required that x < mod and y < mod, but x and y may be
    /// but are not required to be in Montgomery form.
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

#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // sub = x - y; if borrow, add mod back. 1-2 CSR calls.
                // Uses aligned y and mod_ directly as x11.
                alignas(32) UintT res = x;
                uint32_t borrow;
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&y);
                    register uint32_t a2 asm("x12") = 0x02; // SUB
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    borrow = a2;
                }
                if (borrow)
                {
                    // Use aligned mod_ directly as x11 (no copy needed)
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                    register uint32_t a2 asm("x12") = 0x01; // ADD
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }
                return res;
            }
        }
#endif

        const auto d = subc(x, y);
        const auto s = d.value + mod_;
        return (d.carry) ? s : d.value;
    }

    /// Compute the modular inversion of the x in Montgomery form. The result is in Montgomery form.
    /// If x is not invertible, the result is 0.
    constexpr UintT inv(const UintT& x) const noexcept
    {
        assert((mod_ & 1) == 1);
        assert(mod_ >= 3);

        // Precompute inverse of 2 modulo mod: inv2 * 2 % mod == 1.
        // The 1/2 is inexact division that can be fixed by adding "0" to the numerator
        // and making it even: (mod + 1) / 2. To avoid potential overflow of (1 + mod)
        // we rewrite it further to (mod - 1 + 2) / 2 = (mod - 1) / 2 + 1 = ⌊mod / 2⌋ + 1.
        const auto inv2 = (mod_ >> 1) + 1;

        // Use extended binary Euclidean algorithm. This evolves variables a and b until a is 0.
        // Then GCD(x, mod) is in b. If GCD(x, mod) == 1 then the inversion exists and is in v.
        // This follows the classic algorithm (Algorithm 1) presented in
        // "Optimized Binary GCD for Modular Inversion".
        // https://eprint.iacr.org/2020/972.pdf#algorithm.1
        // TODO: The same paper has additional optimizations that could be applied.
        UintT a = x;
        UintT b = mod_;

        // Bézout's coefficients are originally initialized to 1 and 0. But because the input x
        // is in Montgomery form XR the algorithm would compute X⁻¹R⁻¹. To get the expected X⁻¹R,
        // we need to multiply the result by R². We can achieve the same effect "for free"
        // by initializing u to R² instead of 1.
        UintT u = r_squared_;
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
                u = sub(u, v);
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
};
}  // namespace evmmax
