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

/// Declares an aligned, uninitialized buffer and a UintT reference to it.
/// Avoids the zero-initialization overhead of `UintT var;` (whose default
/// constructor zeroes all words due to `uint64_t words_[N]{}` in intx).
/// The CSR MEMCOPY that follows always overwrites the buffer before reading,
/// so the initial contents are irrelevant.
#define DECL_UNINIT_BUF(UintT, name) \
    alignas(32) char name##_raw_[sizeof(UintT)]; \
    auto& name = *reinterpret_cast<UintT*>(name##_raw_)

/// Declares an aligned, uninitialized buffer and copies src into it using CSR MEMCOPY.
/// Saves 12 instructions vs DECL_UNINIT_BUF + word copy (4 CSR insns vs 16 word insns).
/// Both name and src MUST be 32-byte aligned (guaranteed by DECL_UNINIT_BUF / FieldElement).
#define DECL_UNINIT_BUF_COPY(UintT, name, src) \
    DECL_UNINIT_BUF(UintT, name); \
    do { \
        register uintptr_t a0_ asm("x10") = reinterpret_cast<uintptr_t>(&name); \
        register uintptr_t a1_ asm("x11") = reinterpret_cast<uintptr_t>(&(src)); \
        register uint32_t a2_ asm("x12") = 0x80; \
        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2_) : "r"(a0_), "r"(a1_) : "memory"); \
    } while(0)

/// Copy 256 bits between aligned buffers using CSR MEMCOPY (4 insns vs 16 word insns).
/// Both dst and src MUST be 32-byte aligned.
static inline __attribute__((always_inline))
void csr_copy256(void* dst, const void* src) noexcept
{
    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(dst);
    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(src);
    register uint32_t a2 asm("x12") = 0x80;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
}

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
#if defined(AIRBENDER) && defined(__riscv)
class alignas(32) ModArith  // Class-level alignment ensures constexpr instances in .rodata are aligned.
#else
class ModArith
#endif
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
                // m = x * N' mod 2^256, mN_hi = upper(m * N),
                // result = 0 + mN_hi + carry where carry = (x != 0).
                // Then conditional subtract mod.
                //
                // Merged asm blocks: MUL_LOW+MUL_HIGH share x10, ADD+SUB share x10.
                // Zero check uses short-circuit branching for early exit.

                alignas(32) UintT A{};     // t_hi = 0 -> result
                DECL_UNINIT_BUF(UintT, D); // scratch: x copy -> m -> mN_hi
                D = x;  // word copy (cheaper than MEMCOPY CSR)

                // 1-2. MUL_LOW(D, mod_inv) then MUL_HIGH(D, mod) - merged (x10=pD shared)
                {
                    const uintptr_t pD = reinterpret_cast<uintptr_t>(&D);
                    const uintptr_t pModInv = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
                    asm volatile(
                        "mv x10, %[pD]\n\t"
                        "mv x11, %[pModInv]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        // x10 still pD
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        :
                        : [pD] "r"(pD), [pModInv] "r"(pModInv), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
                }

                // 3. carry = (x != 0) - short-circuit branching
                uint32_t low_carry;
                {
                    const auto* xw = reinterpret_cast<const uint32_t*>(&x);
                    uint32_t w = xw[0];
                    if (w == 0) w |= xw[1];
                    if (w == 0) w |= xw[2];
                    if (w == 0) w |= xw[3];
                    if (w == 0) w |= xw[4];
                    if (w == 0) w |= xw[5];
                    if (w == 0) w |= xw[6];
                    if (w == 0) w |= xw[7];
                    low_carry = (w != 0) ? 1u : 0u;
                }

                // 4-6. ADD(A, D+carry) + SUB(A, mod) + conditional ADD - merged
                {
                    const uintptr_t pA = reinterpret_cast<uintptr_t>(&A);
                    const uintptr_t pD = reinterpret_cast<uintptr_t>(&D);
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
                    uint32_t tmp;
                    asm volatile(
                        // ADD(A, D + carry)
                        "mv x10, %[pA]\n\t"
                        "mv x11, %[pD]\n\t"
                        "slli x12, %[carry], 6\n\t"
                        "ori x12, x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "mv %[tmp], x12\n\t"  // tmp = carry out

                        // SUB(A, mod) - x10=pA stays
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Conditional ADD back if carry==0 && borrow!=0
                        // x10=pA, x11=pMod still valid
                        "bnez %[tmp], 2f\n\t"
                        "beqz x12, 2f\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "2:\n\t"

                        : [tmp] "=&r"(tmp)
                        : [pA] "r"(pA), [pD] "r"(pD), [pMod] "r"(pMod),
                          [carry] "r"(low_carry)
                        : "x10", "x11", "x12", "memory"
                    );
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

                DECL_UNINIT_BUF(UintT, A);       // x -> t_lo -> t_hi -> result
                DECL_UNINIT_BUF(UintT, B);       // scratch: t_lo saved -> m -> mn_hi

                // Resolve y pointer: use &y directly if 32-byte aligned, else copy.
                // FieldElement::value_ is alignas(32) so this is the common path.
                DECL_UNINIT_BUF(UintT, Y_buf);
                const bool y_al = (reinterpret_cast<uintptr_t>(&y) % 32 == 0);
                if (!y_al) Y_buf = y;
                const uintptr_t y_ptr = y_al
                    ? reinterpret_cast<uintptr_t>(&y)
                    : reinterpret_cast<uintptr_t>(&Y_buf);

                // 0. Copy x -> A: use MEMCOPY if x is aligned, else word copy.
                const bool x_al = (reinterpret_cast<uintptr_t>(&x) % 32 == 0);
                if (!x_al)
                    A = x;

                if (x_al) {
                    // Aligned fast path: single asm block for all CSR operations.
                    // Avoids redundant pointer reloads between separate asm volatile blocks.
                    const uintptr_t pA = reinterpret_cast<uintptr_t>(&A);
                    const uintptr_t pB = reinterpret_cast<uintptr_t>(&B);
                    const uintptr_t pX = reinterpret_cast<uintptr_t>(&x);
                    const uintptr_t pY = y_ptr;
                    const uintptr_t pModInv = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);

                    uint32_t tmp;
                    asm volatile(
                        // Reordered: all B-ptr ops first, then A-ptr ops.
                        // Keeps x10 stable across consecutive CSR calls, saving 1 mv.

                        // Step 0: MEMCOPY x -> A  (x10=pA, x11=pX)
                        "mv x10, %[pA]\n\t"
                        "mv x11, %[pX]\n\t"
                        "li x12, 0x80\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 1: MEMCOPY x -> B  (x10=pB, x11=pX stays)
                        "mv x10, %[pB]\n\t"
                        // x11 already pX from step 0
                        "li x12, 0x80\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 2: MUL_LOW(B, y) -> B = t_lo  (x10=pB stays, x11=pY)
                        // x10 still pB from step 1
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 3: Zero check on B (short-circuit: first nonzero word -> carry=1)
                        "lw %[tmp], 0(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 4(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 8(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 12(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 16(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 20(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 24(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 28(%[pB])\n\t"
                        "1:\n\t"
                        "snez %[tmp], %[tmp]\n\t"  // tmp = (t_lo != 0) ? 1 : 0

                        // Step 4: MUL_LOW(B, mod_inv) -> B = m  (x10=pB stays, x11=pModInv)
                        "mv x11, %[pModInv]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 5: MUL_HIGH(B, mod) -> B = mN_hi  (x10=pB stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 6: MUL_HIGH(A, y) -> A = t_hi  (x10=pA, x11=pY)
                        // Deferred from before: A still holds x (untouched since step 0).
                        "mv x10, %[pA]\n\t"
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 7: ADD(A, B + carry) -> A += B  (x10=pA stays, x11=pB)
                        // x10 still pA from step 6
                        "mv x11, %[pB]\n\t"
                        "slli x12, %[tmp], 6\n\t"
                        "ori x12, x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "mv %[tmp], x12\n\t"  // tmp = carry out from ADD

                        // Step 8: SUB(A, mod) -> A -= mod  (x10=pA stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 9: Conditional ADD back if carry==0 && borrow!=0
                        // x10=pA, x11=pMod both still valid from step 8
                        "bnez %[tmp], 2f\n\t"   // if carry != 0, skip (result valid)
                        "beqz x12, 2f\n\t"      // if borrow == 0, skip (no underflow)
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "2:\n\t"

                        : [tmp] "=&r"(tmp)
                        : [pA] "r"(pA), [pB] "r"(pB), [pX] "r"(pX),
                          [pY] "r"(pY), [pModInv] "r"(pModInv), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
                } else {
                    // Unaligned x path: A already contains x from word copy above.
                    // Steps 1-9 with word-copy reloads where needed.

                    // 1. MUL_LOW(A, y) -> A = t_lo
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = y_ptr;
                        register uint32_t a2 asm("x12") = 0x08;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }

                    // 2. MEMCOPY A -> B
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&A);
                        register uint32_t a2 asm("x12") = 0x80;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }

                    // 3. Reload x -> A (word copy), then MUL_HIGH
                    A = x;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = y_ptr;
                        register uint32_t a2 asm("x12") = 0x10;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }

                    // 4. Zero check
                    uint32_t low_carry;
                    {
                        const auto* bw = reinterpret_cast<const uint32_t*>(&B);
                        uint32_t w = bw[0];
                        if (w == 0) w |= bw[1];
                        if (w == 0) w |= bw[2];
                        if (w == 0) w |= bw[3];
                        if (w == 0) w |= bw[4];
                        if (w == 0) w |= bw[5];
                        if (w == 0) w |= bw[6];
                        if (w == 0) w |= bw[7];
                        low_carry = (w != 0) ? 1u : 0u;
                    }

                    // 5-9: same as before
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                        register uint32_t a2 asm("x12") = 0x08;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&B);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x10;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                    }
                    uint32_t carry;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&B);
                        register uint32_t a2 asm("x12") = 0x01 | (low_carry << 6);
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                        carry = a2;
                    }
                    uint32_t borrow;
                    {
                        register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                        register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&mod_);
                        register uint32_t a2 asm("x12") = 0x02;
                        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                        borrow = a2;
                    }
                    if (!carry & borrow)
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

    /// In-place Montgomery modular multiplication: x = x * y mod.
    /// Saves one MEMCOPY vs mul() by using x's aligned buffer directly as x10.
    /// Requires x to be 32-byte aligned (guaranteed by FieldElement::value_).
    constexpr void mul_assign(UintT& x, const UintT& y) const noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // CSR requires x10 != x11 AND both 32-byte aligned.
                // If x and y alias, fall back to mul(). If either operand is
                // unaligned (const UintT& from inv chains may have only
                // natural alignment), fall back to out-of-place mul().
                if (&x == &y) { x = mul(x, y); return; }
                const auto px_bits = reinterpret_cast<uintptr_t>(&x) & 31;
                const auto py_bits = reinterpret_cast<uintptr_t>(&y) & 31;
                if ((px_bits | py_bits) != 0) {
                    x = mul(x, y);
                    return;
                }

                DECL_UNINIT_BUF(UintT, B);       // scratch: x_copy -> t_lo -> m -> mN_hi

                {
                    const uintptr_t pX = reinterpret_cast<uintptr_t>(&x);
                    const uintptr_t pB = reinterpret_cast<uintptr_t>(&B);
                    const uintptr_t pY = reinterpret_cast<uintptr_t>(&y);
                    const uintptr_t pModInv = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);

                    uint32_t tmp;
                    asm volatile(
                        // Reordered: all B-ptr ops first, then x-ptr ops.
                        // Keeps x10 stable across consecutive CSR calls, saving 1 mv.

                        // Step 0: MEMCOPY x -> B  (x10=pB, x11=pX)
                        "mv x10, %[pB]\n\t"
                        "mv x11, %[pX]\n\t"
                        "li x12, 0x80\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 1: MUL_LOW(B, y) -> B = t_lo  (x10=pB stays, x11=pY)
                        // x10 still pB from step 0
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 2: Zero check on B (t_lo)
                        "lw %[tmp], 0(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 4(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 8(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 12(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 16(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 20(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 24(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 28(%[pB])\n\t"
                        "1:\n\t"
                        "snez %[tmp], %[tmp]\n\t"

                        // Step 3: MUL_LOW(B, mod_inv) -> B = m  (x10=pB stays, x11=pModInv)
                        "mv x11, %[pModInv]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 4: MUL_HIGH(B, mod) -> B = mN_hi  (x10=pB stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 5: MUL_HIGH(x, y) -> x = t_hi  (x10=pX, x11=pY)
                        // Deferred: x still holds original value (untouched since step 0).
                        "mv x10, %[pX]\n\t"
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 6: ADD(x, B + carry) -> x += B  (x10=pX stays, x11=pB)
                        "mv x11, %[pB]\n\t"
                        "slli x12, %[tmp], 6\n\t"
                        "ori x12, x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "mv %[tmp], x12\n\t"

                        // Step 7: SUB(x, mod)  (x10=pX stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 8: Conditional ADD back
                        "bnez %[tmp], 2f\n\t"
                        "beqz x12, 2f\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "2:\n\t"

                        : [tmp] "=&r"(tmp)
                        : [pX] "r"(pX), [pB] "r"(pB),
                          [pY] "r"(pY), [pModInv] "r"(pModInv), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
                }
                return;
            }
        }
#endif
        x = mul(x, y);
    }

    /// Performs N consecutive modular squarings: x = x^(2^n) mod p.
    /// Keeps result in aligned buffer across iterations to avoid inter-call overhead.
    /// Uses 3 buffers: A (result), B (scratch for t_lo/m/mN), C (holds input copy as y operand).
    /// CSR constraint: x10 != x11 for all BigInt calls.
    constexpr UintT square_n(const UintT& x, unsigned n) const noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated() && n > 0)
            {
                DECL_UNINIT_BUF(UintT, A);       // result accumulator
                DECL_UNINIT_BUF(UintT, B);       // scratch: t_lo -> m -> mN_hi
                DECL_UNINIT_BUF(UintT, C);       // holds copy of input (y operand for squaring)

                // Initial MEMCOPY x → A (x always 32-byte aligned from callers).
                {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
                    register uint32_t a2 asm("x12") = 0x80;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                }

                const uintptr_t pA = reinterpret_cast<uintptr_t>(&A);
                const uintptr_t pB = reinterpret_cast<uintptr_t>(&B);
                const uintptr_t pC = reinterpret_cast<uintptr_t>(&C);
                const uintptr_t pModInv = reinterpret_cast<uintptr_t>(&mod_inv_full_);
                const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);

                for (unsigned iter = 0; iter < n; ++iter)
                {
                    // One squaring: A = A^2 mod p
                    // Steps 0-1: Copy A -> C, Copy A -> B
                    // Step 2: MUL_LOW(B, C) -> B = t_lo
                    // Steps 3-5: zero check, MUL_LOW(B,mod_inv), MUL_HIGH(B,mod)
                    // Step 6: MUL_HIGH(A, C) -> A = t_hi (deferred)
                    // Steps 7-9: ADD, SUB, conditional ADD
                    uint32_t tmp;
                    asm volatile(
                        // Reordered: all B-ptr ops first, then A-ptr ops.
                        // Keeps x10 stable across consecutive CSR calls, saving 1 mv.

                        // Step 0: MEMCOPY A -> C  (x10=pC, x11=pA)
                        "mv x10, %[pC]\n\t"
                        "mv x11, %[pA]\n\t"
                        "li x12, 0x80\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 1: MEMCOPY A -> B  (x10=pB, x11=pA stays)
                        "mv x10, %[pB]\n\t"
                        // x11 already pA from step 0
                        "li x12, 0x80\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 2: MUL_LOW(B, C) -> B = t_lo  (x10=pB stays, x11=pC)
                        // x10 still pB from step 1
                        "mv x11, %[pC]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 3: Zero check on B (t_lo)
                        "lw %[tmp], 0(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 4(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 8(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 12(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 16(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 20(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 24(%[pB])\n\t"
                        "bnez %[tmp], 1f\n\t"
                        "lw %[tmp], 28(%[pB])\n\t"
                        "1:\n\t"
                        "snez %[tmp], %[tmp]\n\t"

                        // Step 4: MUL_LOW(B, mod_inv) -> B = m  (x10=pB stays, x11=pModInv)
                        "mv x11, %[pModInv]\n\t"
                        "li x12, 0x08\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 5: MUL_HIGH(B, mod) -> B = mN_hi  (x10=pB stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 6: MUL_HIGH(A, C) -> A = t_hi  (x10=pA, x11=pC)
                        // Deferred: A still holds original value (untouched since step 0).
                        "mv x10, %[pA]\n\t"
                        "mv x11, %[pC]\n\t"
                        "li x12, 0x10\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 7: ADD(A, B + carry) -> A += B  (x10=pA stays, x11=pB)
                        "mv x11, %[pB]\n\t"
                        "slli x12, %[tmp], 6\n\t"
                        "ori x12, x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "mv %[tmp], x12\n\t"

                        // Step 8: SUB(A, mod) -> A -= mod  (x10=pA stays, x11=pMod)
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        // Step 9: Conditional ADD back
                        // x10=pA, x11=pMod both still valid from step 8
                        "bnez %[tmp], 2f\n\t"
                        "beqz x12, 2f\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "2:\n\t"

                        : [tmp] "=&r"(tmp)
                        : [pA] "r"(pA), [pB] "r"(pB), [pC] "r"(pC),
                          [pModInv] "r"(pModInv), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
                }

                return A;
            }
        }
#endif
        // Fallback: N individual squarings
        UintT result = x;
        for (unsigned i = 0; i < n; ++i)
            result = mul(result, result);
        return result;
    }

#if defined(AIRBENDER) && defined(__riscv)
    /// In-place repeated squaring: x = x^(2^n) mod p.
    /// Avoids the return-value copy overhead of square_n (saves ~24 insns per call
    /// by writing the result back via CSR MEMCOPY instead of word-by-word return copy).
    /// Requires x to be 32-byte aligned (guaranteed by DECL_UNINIT_BUF / FieldElement).
    void square_n_inplace(UintT& x, unsigned n) const noexcept
        requires(UintT::num_bits == 256)
    {
        if (n == 0) return;

        DECL_UNINIT_BUF(UintT, A);       // result accumulator
        DECL_UNINIT_BUF(UintT, B);       // scratch: t_lo -> m -> mN_hi
        DECL_UNINIT_BUF(UintT, C);       // holds copy of input (y operand for squaring)

        // Initial copy x → A. CSR MEMCOPY requires 32-byte alignment on x; fall
        // back to word copy when x is unaligned (can happen for const UintT&
        // parameters from inv chains whose caller has natural alignment).
        if ((reinterpret_cast<uintptr_t>(&x) & 31) == 0)
        {
            register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&A);
            register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
            register uint32_t a2 asm("x12") = 0x80;
            asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
        }
        else
        {
            auto* d = reinterpret_cast<uint32_t*>(&A);
            const auto* s = reinterpret_cast<const uint32_t*>(&x);
            d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; d[3]=s[3];
            d[4]=s[4]; d[5]=s[5]; d[6]=s[6]; d[7]=s[7];
        }

        const uintptr_t pA = reinterpret_cast<uintptr_t>(&A);
        const uintptr_t pB = reinterpret_cast<uintptr_t>(&B);
        const uintptr_t pC = reinterpret_cast<uintptr_t>(&C);
        const uintptr_t pModInv = reinterpret_cast<uintptr_t>(&mod_inv_full_);
        const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);

        for (unsigned iter = 0; iter < n; ++iter)
        {
            uint32_t tmp;
            asm volatile(
                // Step 0: MEMCOPY A -> C  (x10=pC, x11=pA)
                "mv x10, %[pC]\n\t"
                "mv x11, %[pA]\n\t"
                "li x12, 0x80\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 1: MEMCOPY A -> B  (x10=pB, x11=pA stays)
                "mv x10, %[pB]\n\t"
                "li x12, 0x80\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 2: MUL_LOW(B, C) -> B = t_lo  (x10=pB stays, x11=pC)
                "mv x11, %[pC]\n\t"
                "li x12, 0x08\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 3: Zero check on B (t_lo)
                "lw %[tmp], 0(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 4(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 8(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 12(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 16(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 20(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 24(%[pB])\n\t"
                "bnez %[tmp], 1f\n\t"
                "lw %[tmp], 28(%[pB])\n\t"
                "1:\n\t"
                "snez %[tmp], %[tmp]\n\t"

                // Step 4: MUL_LOW(B, mod_inv) -> B = m  (x10=pB stays, x11=pModInv)
                "mv x11, %[pModInv]\n\t"
                "li x12, 0x08\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 5: MUL_HIGH(B, mod) -> B = mN_hi  (x10=pB stays, x11=pMod)
                "mv x11, %[pMod]\n\t"
                "li x12, 0x10\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 6: MUL_HIGH(A, C) -> A = t_hi  (x10=pA, x11=pC)
                "mv x10, %[pA]\n\t"
                "mv x11, %[pC]\n\t"
                "li x12, 0x10\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 7: ADD(A, B + carry) -> A += B  (x10=pA stays, x11=pB)
                "mv x11, %[pB]\n\t"
                "slli x12, %[tmp], 6\n\t"
                "ori x12, x12, 0x01\n\t"
                "csrrw x0, 0x7CA, x0\n\t"
                "mv %[tmp], x12\n\t"

                // Step 8: SUB(A, mod) -> A -= mod  (x10=pA stays, x11=pMod)
                "mv x11, %[pMod]\n\t"
                "li x12, 0x02\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 9: Conditional ADD back
                "bnez %[tmp], 2f\n\t"
                "beqz x12, 2f\n\t"
                "li x12, 0x01\n\t"
                "csrrw x0, 0x7CA, x0\n\t"
                "2:\n\t"

                : [tmp] "=&r"(tmp)
                : [pA] "r"(pA), [pB] "r"(pB), [pC] "r"(pC),
                  [pModInv] "r"(pModInv), [pMod] "r"(pMod)
                : "x10", "x11", "x12", "memory"
            );
        }

        // Copy A -> x at the end using CSR MEMCOPY.
        csr_copy256(&x, &A);
    }
#endif

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
                DECL_UNINIT_BUF(UintT, res);
                // Copy x → res: MEMCOPY if aligned, else word copy.
                if (reinterpret_cast<uintptr_t>(&x) % 32 == 0) {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
                    register uint32_t a2 asm("x12") = 0x80;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                } else { res = x; }
                // Resolve y pointer: use directly if aligned.
                DECL_UNINIT_BUF(UintT, yy_buf);
                const bool y_al = (reinterpret_cast<uintptr_t>(&y) % 32 == 0);
                if (!y_al) yy_buf = y;
                const uintptr_t y_ptr = y_al
                    ? reinterpret_cast<uintptr_t>(&y)
                    : reinterpret_cast<uintptr_t>(&yy_buf);
                // Single asm block: ADD, unconditional SUB mod, conditional ADD back.
                // Saves 2-4 mv instructions vs separate asm blocks (x10/x11 reuse).
                {
                    const uintptr_t pR = reinterpret_cast<uintptr_t>(&res);
                    const uintptr_t pY = y_ptr;
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
                    uint32_t tmp;
                    asm volatile(
                        "mv x10, %[pR]\n\t"
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "mv %[tmp], x12\n\t"

                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        "bnez %[tmp], 1f\n\t"
                        "beqz x12, 1f\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "1:\n\t"

                        : [tmp] "=&r"(tmp)
                        : [pR] "r"(pR), [pY] "r"(pY), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
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
                DECL_UNINIT_BUF(UintT, res);
                if (reinterpret_cast<uintptr_t>(&x) % 32 == 0) {
                    register uintptr_t a0 asm("x10") = reinterpret_cast<uintptr_t>(&res);
                    register uintptr_t a1 asm("x11") = reinterpret_cast<uintptr_t>(&x);
                    register uint32_t a2 asm("x12") = 0x80;
                    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2) : "r"(a0), "r"(a1) : "memory");
                } else { res = x; }
                DECL_UNINIT_BUF(UintT, yy_buf);
                const bool y_al = (reinterpret_cast<uintptr_t>(&y) % 32 == 0);
                if (!y_al) yy_buf = y;
                const uintptr_t y_ptr = y_al
                    ? reinterpret_cast<uintptr_t>(&y)
                    : reinterpret_cast<uintptr_t>(&yy_buf);
                // Single asm block: SUB, then conditional ADD mod back.
                // Saves 1-2 mv instructions vs separate asm blocks (x10 reuse).
                {
                    const uintptr_t pR = reinterpret_cast<uintptr_t>(&res);
                    const uintptr_t pY = y_ptr;
                    const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
                    asm volatile(
                        "mv x10, %[pR]\n\t"
                        "mv x11, %[pY]\n\t"
                        "li x12, 0x02\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"

                        "beqz x12, 1f\n\t"
                        "mv x11, %[pMod]\n\t"
                        "li x12, 0x01\n\t"
                        "csrrw x0, 0x7CA, x0\n\t"
                        "1:\n\t"

                        :
                        : [pR] "r"(pR), [pY] "r"(pY), [pMod] "r"(pMod)
                        : "x10", "x11", "x12", "memory"
                    );
                }
                return res;
            }
        }
#endif

        const auto d = subc(x, y);
        const auto s = d.value + mod_;
        return (d.carry) ? s : d.value;
    }

#if defined(AIRBENDER) && defined(__riscv)
    /// In-place modular addition: x += y (mod p).
    /// Requires x and y to be 32-byte aligned (FieldElement::value_ always is).
    /// Saves one MEMCOPY CSR call vs the out-of-place add().
    void __attribute__((always_inline)) add_assign(UintT& x, const UintT& y) const noexcept
        requires(UintT::num_bits == 256)
    {
        // CSR requires x10 != x11 AND both 32-byte aligned. x is alignas(32),
        // but y may be an unaligned const UintT& from inv chains. If y is not
        // 32-aligned, copy it to an aligned scratch (same path as self-alias).
        // Self-add (x += x): copy via CSR MEMCOPY to yy_buf (1 insn vs word copy).
        DECL_UNINIT_BUF(UintT, yy_buf);
        const bool y_aliased = (&x == &y);
        const bool y_unaligned = (reinterpret_cast<uintptr_t>(&y) & 31) != 0;
        const bool y_needs_copy = y_aliased || y_unaligned;
        if (y_aliased) {
            register uintptr_t a0_ asm("x10") = reinterpret_cast<uintptr_t>(&yy_buf);
            register uintptr_t a1_ asm("x11") = reinterpret_cast<uintptr_t>(&y);
            register uint32_t a2_ asm("x12") = 0x80;
            asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2_) : "r"(a0_), "r"(a1_) : "memory");
        } else if (y_unaligned) {
            // Word copy from unaligned y into aligned yy_buf (8 word stores).
            auto* d_ = reinterpret_cast<uint32_t*>(&yy_buf);
            const auto* s_ = reinterpret_cast<const uint32_t*>(&y);
            d_[0]=s_[0]; d_[1]=s_[1]; d_[2]=s_[2]; d_[3]=s_[3];
            d_[4]=s_[4]; d_[5]=s_[5]; d_[6]=s_[6]; d_[7]=s_[7];
        }
        const uintptr_t y_ptr = y_needs_copy
            ? reinterpret_cast<uintptr_t>(&yy_buf)
            : reinterpret_cast<uintptr_t>(&y);
        // Single asm block: ADD, then unconditional SUB mod, then conditional ADD back.
        // Saves 2-4 mv instructions vs separate asm blocks (x10/x11 reuse across CSR ops).
        {
            const uintptr_t pX = reinterpret_cast<uintptr_t>(&x);
            const uintptr_t pY = y_ptr;
            const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
            uint32_t tmp;
            asm volatile(
                // Step 0: ADD(x, y) -> x += y
                "mv x10, %[pX]\n\t"
                "mv x11, %[pY]\n\t"
                "li x12, 0x01\n\t"
                "csrrw x0, 0x7CA, x0\n\t"
                "mv %[tmp], x12\n\t"    // tmp = carry from ADD

                // Step 1: SUB(x, mod) -> x -= mod  (x10=pX stays)
                "mv x11, %[pMod]\n\t"
                "li x12, 0x02\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 2: Conditional ADD back if !carry && borrow
                // x10=pX, x11=pMod both still valid from step 1
                "bnez %[tmp], 1f\n\t"   // if carry != 0, SUB was correct
                "beqz x12, 1f\n\t"      // if borrow == 0, SUB was correct
                "li x12, 0x01\n\t"
                "csrrw x0, 0x7CA, x0\n\t"
                "1:\n\t"

                : [tmp] "=&r"(tmp)
                : [pX] "r"(pX), [pY] "r"(pY), [pMod] "r"(pMod)
                : "x10", "x11", "x12", "memory"
            );
        }
    }

    /// In-place modular subtraction: x -= y (mod p).
    /// Requires x and y to be 32-byte aligned.
    void __attribute__((always_inline)) sub_assign(UintT& x, const UintT& y) const noexcept
        requires(UintT::num_bits == 256)
    {
        // CSR requires x10 != x11 AND both 32-byte aligned. x is alignas(32),
        // but y may be an unaligned const UintT& from inv chains. If y is not
        // 32-aligned, copy it to an aligned scratch (same path as self-alias).
        DECL_UNINIT_BUF(UintT, yy_buf);
        const bool y_aliased = (&x == &y);
        const bool y_unaligned = (reinterpret_cast<uintptr_t>(&y) & 31) != 0;
        const bool y_needs_copy = y_aliased || y_unaligned;
        if (y_aliased) {
            register uintptr_t a0_ asm("x10") = reinterpret_cast<uintptr_t>(&yy_buf);
            register uintptr_t a1_ asm("x11") = reinterpret_cast<uintptr_t>(&y);
            register uint32_t a2_ asm("x12") = 0x80;
            asm volatile("csrrw x0, 0x7CA, x0" : "+r"(a2_) : "r"(a0_), "r"(a1_) : "memory");
        } else if (y_unaligned) {
            auto* d_ = reinterpret_cast<uint32_t*>(&yy_buf);
            const auto* s_ = reinterpret_cast<const uint32_t*>(&y);
            d_[0]=s_[0]; d_[1]=s_[1]; d_[2]=s_[2]; d_[3]=s_[3];
            d_[4]=s_[4]; d_[5]=s_[5]; d_[6]=s_[6]; d_[7]=s_[7];
        }
        const uintptr_t y_ptr = y_needs_copy
            ? reinterpret_cast<uintptr_t>(&yy_buf)
            : reinterpret_cast<uintptr_t>(&y);
        // Single asm block: SUB, then conditional ADD mod back.
        // Saves 1-2 mv instructions vs separate asm blocks (x10 reuse across CSR ops).
        {
            const uintptr_t pX = reinterpret_cast<uintptr_t>(&x);
            const uintptr_t pY = y_ptr;
            const uintptr_t pMod = reinterpret_cast<uintptr_t>(&mod_);
            asm volatile(
                // Step 0: SUB(x, y) -> x -= y
                "mv x10, %[pX]\n\t"
                "mv x11, %[pY]\n\t"
                "li x12, 0x02\n\t"
                "csrrw x0, 0x7CA, x0\n\t"

                // Step 1: Conditional ADD(x, mod) if borrow
                // x10=pX stays from step 0
                "beqz x12, 1f\n\t"      // if borrow == 0, skip ADD
                "mv x11, %[pMod]\n\t"
                "li x12, 0x01\n\t"
                "csrrw x0, 0x7CA, x0\n\t"
                "1:\n\t"

                :
                : [pX] "r"(pX), [pY] "r"(pY), [pMod] "r"(pMod)
                : "x10", "x11", "x12", "memory"
            );
        }
    }
#endif

    /// Optimized BN254 Fp Fermat inversion: x^(p-2) mod p.
    /// p-2 = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45
    /// Uses sliding window w=5: 252S + 54M = 306 Montgomery muls (vs generic 362).
    /// Marked noinline to prevent flatten from inlining all 54 mul calls (262KB code bloat).
    __attribute__((noinline)) UintT inv_bn254_fp(const UintT& x) const noexcept
    {
        // Precomputation: x^2 and odd powers x^3..x^31
        // CSR MEMCOPY copy (4 insns) + mul_assign; DECL_UNINIT_BUF avoids zero-init
        DECL_UNINIT_BUF_COPY(UintT, x2, x); mul_assign(x2, x);
        DECL_UNINIT_BUF_COPY(UintT, x3, x); mul_assign(x3, x2);
        DECL_UNINIT_BUF_COPY(UintT, x5, x3); mul_assign(x5, x2);
        DECL_UNINIT_BUF_COPY(UintT, x7, x5); mul_assign(x7, x2);
        DECL_UNINIT_BUF_COPY(UintT, x9, x7); mul_assign(x9, x2);
        DECL_UNINIT_BUF_COPY(UintT, x11, x9); mul_assign(x11, x2);
        DECL_UNINIT_BUF_COPY(UintT, x13, x11); mul_assign(x13, x2);
        DECL_UNINIT_BUF_COPY(UintT, x15, x13); mul_assign(x15, x2);
        DECL_UNINIT_BUF_COPY(UintT, x17, x15); mul_assign(x17, x2);
        DECL_UNINIT_BUF_COPY(UintT, x19, x17); mul_assign(x19, x2);
        DECL_UNINIT_BUF_COPY(UintT, x21, x19); mul_assign(x21, x2);
        DECL_UNINIT_BUF_COPY(UintT, x23, x21); mul_assign(x23, x2);
        DECL_UNINIT_BUF_COPY(UintT, x25, x23); mul_assign(x25, x2);
        DECL_UNINIT_BUF_COPY(UintT, x27, x25); mul_assign(x27, x2);
        DECL_UNINIT_BUF_COPY(UintT, x29, x27); mul_assign(x29, x2);
        DECL_UNINIT_BUF_COPY(UintT, x31, x29); mul_assign(x31, x2);
        // 16M precomputation

        // Sliding window chain: 252S + 38M
        // square_n_inplace avoids return-value copy overhead (~24 insns/call).
        DECL_UNINIT_BUF_COPY(UintT, r, x3);                  // initial
        square_n_inplace(r, 10); mul_assign(r, x25);   // 10S+1M
        square_n_inplace(r, 8); mul_assign(r, x19);    // 8S+1M
        square_n_inplace(r, 5); mul_assign(r, x19);    // 5S+1M
        square_n_inplace(r, 4); mul_assign(r, x9);     // 4S+1M
        square_n_inplace(r, 4); mul_assign(r, x7);     // 4S+1M
        square_n_inplace(r, 9); mul_assign(r, x19);    // 9S+1M
        square_n_inplace(r, 7); mul_assign(r, x13);    // 7S+1M
        square_n_inplace(r, 10); mul_assign(r, x5);    // 10S+1M
        square_n_inplace(r, 7); mul_assign(r, x27);    // 7S+1M
        square_n_inplace(r, 1); mul_assign(r, x);      // 1S+1M
        square_n_inplace(r, 7); mul_assign(r, x5);     // 7S+1M
        square_n_inplace(r, 10); mul_assign(r, x17);   // 10S+1M
        square_n_inplace(r, 6); mul_assign(r, x27);    // 6S+1M
        square_n_inplace(r, 5); mul_assign(r, x13);    // 5S+1M
        square_n_inplace(r, 8); mul_assign(r, x3);     // 8S+1M
        square_n_inplace(r, 11); mul_assign(r, x21);   // 11S+1M
        square_n_inplace(r, 1); mul_assign(r, x);      // 1S+1M
        square_n_inplace(r, 9); mul_assign(r, x23);    // 9S+1M
        square_n_inplace(r, 6); mul_assign(r, x25);    // 6S+1M
        square_n_inplace(r, 5); mul_assign(r, x15);    // 5S+1M
        square_n_inplace(r, 10); mul_assign(r, x11);   // 10S+1M
        square_n_inplace(r, 6); mul_assign(r, x21);    // 6S+1M
        square_n_inplace(r, 7); mul_assign(r, x17);    // 7S+1M
        square_n_inplace(r, 5); mul_assign(r, x13);    // 5S+1M
        square_n_inplace(r, 7); mul_assign(r, x7);     // 7S+1M
        square_n_inplace(r, 6); mul_assign(r, x7);     // 6S+1M
        square_n_inplace(r, 7); mul_assign(r, x21);    // 7S+1M
        square_n_inplace(r, 7); mul_assign(r, x13);    // 7S+1M
        square_n_inplace(r, 6); mul_assign(r, x15);    // 6S+1M
        square_n_inplace(r, 5); mul_assign(r, x);      // 5S+1M
        square_n_inplace(r, 10); mul_assign(r, x17);   // 10S+1M
        square_n_inplace(r, 1); mul_assign(r, x);      // 1S+1M
        square_n_inplace(r, 9); mul_assign(r, x11);    // 9S+1M
        square_n_inplace(r, 6); mul_assign(r, x27);    // 6S+1M
        square_n_inplace(r, 9); mul_assign(r, x31);    // 9S+1M
        square_n_inplace(r, 7); mul_assign(r, x31);    // 7S+1M
        square_n_inplace(r, 5); mul_assign(r, x21);    // 5S+1M
        square_n_inplace(r, 6); mul_assign(r, x5);     // 6S+1M
        // Total: 252S + 54M = 306 Montgomery muls
        return r;
    }

    /// Compute the modular inversion of the x in Montgomery form. The result is in Montgomery form.
    /// If x is not invertible, the result is 0.
    constexpr __attribute__((flatten)) UintT inv(const UintT& x) const noexcept
    {
#if defined(AIRBENDER) && defined(__riscv)
        if constexpr (UintT::num_bits == 256)
        {
            if (!std::is_constant_evaluated())
            {
                // Fermat inversion: x^{mod-2} mod p, using CSR-accelerated Montgomery muls.
                // For the secp256k1 field prime, uses an optimized addition chain (266S+14M=280 muls).
                // For other primes, uses generic square-and-multiply (~383 muls).
                // Both are faster than the binary GCD (~512 iterations of non-CSR ops).

                if (x == UintT{0}) [[unlikely]]
                    return UintT{0};

                // secp256k1 field prime p
                constexpr UintT SECP256K1_P =
                    intx::from_string<UintT>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

                if (mod_ == SECP256K1_P)
                {
                    // Optimized addition chain for p-2 exponent.
                    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
                    //     = 2^256 - 2^32 - 979
                    // Reuses the field_sqrt chain (253S+13M) up to x223, then 34S+3M for tail.
                    // Total: 266S + 14M = 280 Montgomery muls.

                    // Use uninit buffers to avoid dead zero-init of 6 uint256 vars (48 sw zero).
                    DECL_UNINIT_BUF(UintT, z);
                    DECL_UNINIT_BUF(UintT, t0);
                    DECL_UNINIT_BUF(UintT, t1);
                    DECL_UNINIT_BUF(UintT, t2);
                    DECL_UNINIT_BUF(UintT, t3);
                    DECL_UNINIT_BUF(UintT, f);

                    // Step 1: z = x^0x2
                    csr_copy256(&z, &x); mul_assign(z, x);   // CSR copy+mul_assign
                    // Step 2: z = x^0x3
                    mul_assign(z, x);
                    // Step 4: t0 = x^0xc (2 squarings of z)
                    csr_copy256(&t0, &z); square_n_inplace(t0, 2);
                    // Step 5: t0 = x^0xf
                    mul_assign(t0, z);
                    // Save x^15 for computing x^45 later
                    csr_copy256(&f, &t0);
                    // Step 6: t1 = x^0x1e
                    csr_copy256(&t1, &t0); mul_assign(t1, t0);  // CSR copy+mul_assign
                    // Step 7: t2 = x^0x1f
                    csr_copy256(&t2, &t1); mul_assign(t2, x);   // CSR copy+mul_assign
                    // Step 9: t1 = x^0x7c (2 squarings of t2)
                    csr_copy256(&t1, &t2); square_n_inplace(t1, 2);
                    // Step 10: t1 = x^0x7f
                    mul_assign(t1, z);
                    // Step 14: t3 = x^0x7f0 (4 squarings of t1)
                    csr_copy256(&t3, &t1); square_n_inplace(t3, 4);
                    // Step 15: t0 = x^0x7ff
                    mul_assign(t0, t3);
                    // Step 26: t3 = x^0x3ff800 (11 squarings of t0)
                    csr_copy256(&t3, &t0); square_n_inplace(t3, 11);
                    // Step 27: t0 = x^0x3fffff  (x22 = x^{2^22-1})
                    mul_assign(t0, t3);
                    // Step 32: t3 = x^0x7ffffe0 (5 squarings of t0)
                    csr_copy256(&t3, &t0); square_n_inplace(t3, 5);
                    // Step 33: t2 = x^0x7ffffff  (x27)
                    mul_assign(t2, t3);
                    // Step 60: t3 = x^0x3ffffff8000000 (27 squarings of t2)
                    csr_copy256(&t3, &t2); square_n_inplace(t3, 27);
                    // Step 61: t2 = x^0x3fffffffffffff  (x54)
                    mul_assign(t2, t3);
                    // Step 115: t3 = (54 squarings of t2)
                    csr_copy256(&t3, &t2); square_n_inplace(t3, 54);
                    // Step 116: t2 = x^0xfffffffffffffffffffffffffff  (x108)
                    mul_assign(t2, t3);
                    // Step 224: t3 = (108 squarings of t2)
                    csr_copy256(&t3, &t2); square_n_inplace(t3, 108);
                    // Step 225: t2 = x^{2^216-1}  (x216)
                    mul_assign(t2, t3);
                    // Step 232: t2 = x^{(2^216-1)*2^7} (7 squarings)
                    square_n_inplace(t2, 7);
                    // Step 233: t1 = x^{2^223-1}  (x223)
                    mul_assign(t1, t2);

                    // --- Tail for p-2 ---
                    // Step 256: t1 = x^{(2^223-1)*2^23} (23 squarings)
                    square_n_inplace(t1, 23);
                    // Step 257: t0 = x^{2^246 - 2^22 - 1}
                    mul_assign(t0, t1);
                    // Step 267: t0 = x^{2^256 - 2^32 - 2^10} (10 squarings)
                    square_n_inplace(t0, 10);
                    // x^45 = (x^15)^3 from saved f
                    csr_copy256(&t3, &f); mul_assign(t3, f);    // x^30 (CSR copy+mul_assign)
                    mul_assign(t3, f);  // x^45
                    // x^{p-2} = x^{2^256-2^32-979}
                    mul_assign(t0, t3);           // in-place saves 1 MEMCOPY vs return mul
                    return t0;
                }

                // secp256k1 scalar field order N
                constexpr UintT SECP256K1_N =
                    intx::from_string<UintT>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

                if (mod_ == SECP256K1_N)
                {
                    // Optimized addition chain for N-2 exponent.
                    // N-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD036413F
                    // Uses hybrid approach: doubling chain for top 125 all-1 bits,
                    // then sliding window w=5 for remaining 131 bits.
                    // Total: 251S + 46M = 297 Montgomery muls (vs generic 450).

                    // Precomputation: x^2 and odd powers x^3..x^31
                    // CSR MEMCOPY copy (4 insns) + mul_assign; DECL_UNINIT_BUF avoids zero-init
                    DECL_UNINIT_BUF_COPY(UintT, x2, x); mul_assign(x2, x);
                    DECL_UNINIT_BUF_COPY(UintT, x3, x); mul_assign(x3, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x5, x3); mul_assign(x5, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x7, x5); mul_assign(x7, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x9, x7); mul_assign(x9, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x11, x9); mul_assign(x11, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x13, x11); mul_assign(x13, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x15, x13); mul_assign(x15, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x17, x15); mul_assign(x17, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x19, x17); mul_assign(x19, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x21, x19); mul_assign(x21, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x23, x21); mul_assign(x23, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x25, x23); mul_assign(x25, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x27, x25); mul_assign(x27, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x29, x27); mul_assign(x29, x2);
                    DECL_UNINIT_BUF_COPY(UintT, x31, x29); mul_assign(x31, x2);

                    // Phase 1: Build x^(2^125-1) via doubling chain from x^31.
                    // x^(2^5-1) = x^31 (from precomp)
                    DECL_UNINIT_BUF_COPY(UintT, r, x31);
                    DECL_UNINIT_BUF(UintT, t);  // avoid dead zero-init
                    // x^(2^10-1) = sq5(x^31) * x^31
                    csr_copy256(&t, &r); square_n_inplace(t, 5);
                    mul_assign(t, x31);
                    DECL_UNINIT_BUF_COPY(UintT, x10_1, t);
                    // x^(2^20-1) = sq10(x^(2^10-1)) * x^(2^10-1)
                    csr_copy256(&t, &x10_1); square_n_inplace(t, 10);
                    mul_assign(t, x10_1);
                    DECL_UNINIT_BUF_COPY(UintT, x20_1, t);
                    // x^(2^25-1) = sq5(x^(2^20-1)) * x^31
                    csr_copy256(&t, &x20_1); square_n_inplace(t, 5);
                    mul_assign(t, x31);
                    DECL_UNINIT_BUF_COPY(UintT, x25_1, t);
                    // x^(2^50-1) = sq25(x^(2^25-1)) * x^(2^25-1)
                    csr_copy256(&t, &x25_1); square_n_inplace(t, 25);
                    mul_assign(t, x25_1);
                    DECL_UNINIT_BUF_COPY(UintT, x50_1, t);
                    // x^(2^100-1) = sq50(x^(2^50-1)) * x^(2^50-1)
                    csr_copy256(&t, &x50_1); square_n_inplace(t, 50);
                    mul_assign(t, x50_1);
                    DECL_UNINIT_BUF_COPY(UintT, x100_1, t);
                    // x^(2^125-1) = sq25(x^(2^100-1)) * x^(2^25-1)
                    csr_copy256(&t, &x100_1); square_n_inplace(t, 25);
                    mul_assign(t, x25_1);
                    csr_copy256(&r, &t);
                    // r = x^(2^125-1), cost: 120S + 6M

                    // Phase 2: Remaining 131 bits of N-2 via sliding window.
                    // square_n_inplace avoids return-value copy overhead (~24 insns/call).
                    // N-2 remaining after top 125 ones:
                    // 11_0_10111010101011101101110011100110...10011111
                    square_n_inplace(r, 4); mul_assign(r, x13);   // 4S+1M
                    square_n_inplace(r, 6); mul_assign(r, x29);   // 6S+1M
                    square_n_inplace(r, 6); mul_assign(r, x21);   // 6S+1M
                    square_n_inplace(r, 5); mul_assign(r, x27);   // 5S+1M
                    square_n_inplace(r, 4); mul_assign(r, x7);    // 4S+1M
                    square_n_inplace(r, 5); mul_assign(r, x7);    // 5S+1M
                    square_n_inplace(r, 6); mul_assign(r, x13);   // 6S+1M
                    square_n_inplace(r, 6); mul_assign(r, x23);   // 6S+1M
                    square_n_inplace(r, 3); mul_assign(r, x5);    // 3S+1M
                    square_n_inplace(r, 7); mul_assign(r, x17);   // 7S+1M
                    square_n_inplace(r, 2); mul_assign(r, x);     // 2S+1M
                    square_n_inplace(r, 12); mul_assign(r, x29);  // 12S+1M
                    square_n_inplace(r, 5); mul_assign(r, x27);   // 5S+1M
                    square_n_inplace(r, 5); mul_assign(r, x31);   // 5S+1M
                    square_n_inplace(r, 3); mul_assign(r, x5);    // 3S+1M
                    square_n_inplace(r, 6); mul_assign(r, x9);    // 6S+1M
                    square_n_inplace(r, 5); mul_assign(r, x15);   // 5S+1M
                    square_n_inplace(r, 6); mul_assign(r, x17);   // 6S+1M
                    square_n_inplace(r, 5); mul_assign(r, x19);   // 5S+1M
                    square_n_inplace(r, 2); mul_assign(r, x);     // 2S+1M
                    square_n_inplace(r, 11); mul_assign(r, x27);  // 11S+1M
                    square_n_inplace(r, 3); mul_assign(r, x);     // 3S+1M
                    square_n_inplace(r, 10); mul_assign(r, x19);  // 10S+1M
                    square_n_inplace(r, 4); mul_assign(r, x15);   // 4S+1M
                    // Total phase 2: 131S + 24M
                    return r;
                }
                // BN254 base field prime (Fp)
                constexpr UintT BN254_P =
                    intx::from_string<UintT>("0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

                if (mod_ == BN254_P)
                    return inv_bn254_fp(x);

                // Generic Fermat: square-and-multiply over bits of (mod - 2).
                {
                    const UintT exp = mod_ - 2;
                    const auto bw = intx::bit_width(exp);
                    UintT result = x;
                    for (size_t i = bw - 1; i != 0; --i)
                    {
                        result = mul(result, result);  // squaring: x==y alias, can't use mul_assign
                        if (intx::bit_test(exp, i - 1))
                            mul_assign(result, x);
                    }
                    return result;
                }
            }
        }
#endif

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
