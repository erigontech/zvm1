#!/bin/sh
# Copyright 2026 The zvm1 Authors (modifications)
#
# Patch blst's vect.h, fields.h and no_asm.h for SP1 BLS12-381 syscalls.
#
# THREE-PHASE PATCHING:
#
# Phase 0 (vect.h + no_asm.h): Force 64-bit limbs on rv64im.
#   blst's vect.h forces limb_t = uint32_t when __BLST_NO_ASM__ is defined
#   (which it is on RISC-V). This makes vec384 = uint32_t[12] with only
#   4-byte alignment. SP1 syscalls require 8-byte alignment, so the old
#   approach used memcpy to/from aligned buffers (~53M cycles overhead on
#   blob-heavy blocks).
#   Fix: Override limb_t to uint64_t on rv64im so vec384 = uint64_t[6]
#   with native 8-byte alignment. Also add __int128 llimb_t to no_asm.h.
#   ALSO: Fix quot_rem_n's __builtin_assume(n%2==0) -- with 64-bit limbs
#   NLIMBS(64)=1 which violates this, causing __builtin_unreachable() UB.
#
# Phase 1 (fields.h): Patches high-level add_fp/sub_fp/add_fp2/sub_fp2/sqr_fp2
#   with direct syscall calls. No memcpy needed since data is now uint64_t[6].
#
# Phase 2 (no_asm.h): Patches lowest-level Montgomery routines:
#   mul_mont_384, sqr_mont_384, redc_mont_384, from_mont_384,
#   sqr_n_mul_mont_383, mul_mont_384x, sqr_mont_382x.
#   This catches ALL callers including exp_mont_384, recip.c,
#   hash_to_field.c, fp12_tower.c, blst_t.hpp, vect.c, etc.
#   No memcpy needed since limb_t is now uint64_t (8-byte aligned).
#
# The SP1 syscalls perform PLAIN modular arithmetic (not Montgomery).
# mul/sqr need an extra R^{-1} correction because blst uses Montgomery form.
set -e

VECT_H="src/vect.h"
FIELDS_H="src/fields.h"
NO_ASM_H="src/no_asm.h"
[ -f "$VECT_H"   ] || { echo "ERR: $VECT_H not found (pwd=$(pwd))"; exit 1; }
[ -f "$FIELDS_H" ] || { echo "ERR: $FIELDS_H not found (pwd=$(pwd))"; exit 1; }
[ -f "$NO_ASM_H" ] || { echo "ERR: $NO_ASM_H not found (pwd=$(pwd))"; exit 1; }

grep -q "SP1_BLS12381_SYSCALLS" "$FIELDS_H" && { echo "Already patched"; exit 0; }

# ────────── Phase 0: Patch vect.h + no_asm.h for 64-bit limbs ──────────
# In vect.h, the __BLST_NO_ASM__ branch forces 32-bit limbs. On rv64im
# (where __riscv_xlen == 64), override to 64-bit limbs.
# In no_asm.h, llimb_t is only defined for 32-bit limbs. Add __int128 for 64-bit.
# Also fix quot_rem_n: with 64-bit limbs NLIMBS(64)=1 which violates the
# n%2==0 __builtin_assume, causing __builtin_unreachable() and UB.

python3 - "$VECT_H" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Replace the __BLST_NO_ASM__ block to check for rv64 first.
# Original:
#   #elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
#   typedef unsigned int limb_t;
#   # define LIMB_T_BITS    32
#   # ifndef __BLST_NO_ASM__
#   #  define __BLST_NO_ASM__
#   # endif
#
# New: on rv64im (__riscv + __riscv_xlen==64), use 64-bit limbs even with __BLST_NO_ASM__.

old_block = """#elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
typedef unsigned int limb_t;
# define LIMB_T_BITS    32
# ifndef __BLST_NO_ASM__
#  define __BLST_NO_ASM__
# endif"""

new_block = """#elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
# if defined(__riscv) && (__riscv_xlen == 64)
/* rv64im: use 64-bit limbs for native 8-byte alignment (SP1 syscall compat) */
typedef unsigned long long limb_t;
#  define LIMB_T_BITS    64
# else
typedef unsigned int limb_t;
#  define LIMB_T_BITS    32
# endif
# ifndef __BLST_NO_ASM__
#  define __BLST_NO_ASM__
# endif"""

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("Patched vect.h: 64-bit limbs on rv64im")
else:
    print("WARNING: could not find limb_t block in vect.h", file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(src)
PYEOF

python3 - "$NO_ASM_H" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Add llimb_t = unsigned __int128 for 64-bit limbs.
# Original:
#   #if LIMB_T_BITS==32
#   typedef unsigned long long llimb_t;
#   #endif
#
# New: also handle LIMB_T_BITS==64 with __int128.

old_llimb = """#if LIMB_T_BITS==32
typedef unsigned long long llimb_t;
#endif"""

new_llimb = """#if LIMB_T_BITS==32
typedef unsigned long long llimb_t;
#elif LIMB_T_BITS==64
typedef unsigned __int128 llimb_t;
#endif"""

if old_llimb in src:
    src = src.replace(old_llimb, new_llimb, 1)
    print("Patched no_asm.h: added __int128 llimb_t for 64-bit limbs")
else:
    print("WARNING: could not find llimb_t block in no_asm.h", file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(src)
PYEOF

# ────────── Phase 1: Patch fields.h (high-level Fp/Fp2 wrappers) ──────────
# With 64-bit limbs, vec384 = limb_t[6] = uint64_t[6] and is natively
# 8-byte aligned. We can pass pointers directly to syscalls -- no memcpy needed.
#
# OPTIMIZATIONS vs previous version:
#  - No function-pointer indirection (_sp1_fp_binop removed) -- direct inline syscalls
#  - Smarter aliasing: only copy when ret overlaps b, not unconditionally
#  - add_fp/sub_fp/add_fp2/sub_fp2 directly inlined with minimal copies
python3 - "$FIELDS_H" << 'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# With 64-bit limbs, vec384 IS uint64_t[6] (natively 8-byte aligned).
# We can cast directly to (unsigned long long *) without memcpy.
decls = r"""
/* ── SP1 BLS12-381 syscall acceleration ── */
#ifdef SP1_BLS12381_SYSCALLS
/* Inline ecalls: bypass Rust wrappers, emit ecall directly. */
#define _SP1_ECALL_FP(name, num)                                        \
    static inline __attribute__((always_inline)) void name(             \
        unsigned long long *_p, const unsigned long long *_q) {         \
        register unsigned long long t0 __asm__("t0") = (num);          \
        register unsigned long long *a0 __asm__("a0") = _p;            \
        register const unsigned long long *a1 __asm__("a1") = _q;      \
        __asm__ volatile("ecall" : "+r"(t0) : "r"(a0), "r"(a1) : "memory"); \
    }
_SP1_ECALL_FP(syscall_bls12381_fp_addmod,  0x00010120)
_SP1_ECALL_FP(syscall_bls12381_fp_submod,  0x00010121)
_SP1_ECALL_FP(syscall_bls12381_fp_mulmod,  0x00010122)
_SP1_ECALL_FP(syscall_bls12381_fp2_addmod, 0x00010123)
_SP1_ECALL_FP(syscall_bls12381_fp2_submod, 0x00010124)
_SP1_ECALL_FP(syscall_bls12381_fp2_mulmod, 0x00010125)

/* R_INV = (2^384)^{-1} mod P, stored as limb_t[6] (natively 8-byte aligned on rv64). */
static const unsigned long long _SP1_FP_R_INV[6] = {
    0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
    0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL
};
/* For Fp2: (R^{-1}, 0) */
static const unsigned long long _SP1_FP2_R_INV[12] = {
    0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
    0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL
};
#endif
"""

src = src.replace('#include "consts.h"', '#include "consts.h"\n' + decls, 1)

# ── Direct inline replacement for add_fp ──
# SP1 syscalls read both operands atomically before writing result.
# So p==q is safe. Only dangerous case: ret overlaps b but not a.
# Use a tmp only when needed.
old_add_fp = """static inline void add_fp(vec384 ret, const vec384 a, const vec384 b)
{   add_mod_384(ret, a, b, BLS12_381_P);   }"""

new_add_fp = """static inline void add_fp(vec384 ret, const vec384 a, const vec384 b)
{
#ifdef SP1_BLS12381_SYSCALLS
    if (ret == a) {
        /* In-place: no copies needed */
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* Commutative: a+b = b+a, ret already holds b */
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)a);
    } else {
        /* ret doesn't overlap either operand */
        vec_copy(ret, a, sizeof(vec384));
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    }
#else
    add_mod_384(ret, a, b, BLS12_381_P);
#endif
}"""

src = src.replace(old_add_fp, new_add_fp, 1)
print("Patched add_fp")

# ── Direct inline replacement for sub_fp ──
old_sub_fp = """static inline void sub_fp(vec384 ret, const vec384 a, const vec384 b)
{   sub_mod_384(ret, a, b, BLS12_381_P);   }"""

new_sub_fp = """static inline void sub_fp(vec384 ret, const vec384 a, const vec384 b)
{
#ifdef SP1_BLS12381_SYSCALLS
    if (ret == a) {
        syscall_bls12381_fp_submod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* ret aliases b: need tmp because sub is non-commutative */
        vec384 tmp;
        vec_copy(tmp, a, sizeof(vec384));
        syscall_bls12381_fp_submod((unsigned long long *)tmp, (const unsigned long long *)b);
        vec_copy(ret, tmp, sizeof(vec384));
    } else {
        /* ret is separate from both: copy a->ret, subtract in-place */
        vec_copy(ret, a, sizeof(vec384));
        syscall_bls12381_fp_submod((unsigned long long *)ret, (const unsigned long long *)b);
    }
#else
    sub_mod_384(ret, a, b, BLS12_381_P);
#endif
}"""

src = src.replace(old_sub_fp, new_sub_fp, 1)
print("Patched sub_fp")

# ── Direct inline replacement for add_fp2 ──
old_add_fp2 = """static inline void add_fp2(vec384x ret, const vec384x a, const vec384x b)
{   add_mod_384x(ret, a, b, BLS12_381_P);   }"""

new_add_fp2 = """static inline void add_fp2(vec384x ret, const vec384x a, const vec384x b)
{
#ifdef SP1_BLS12381_SYSCALLS
    if (ret == a) {
        syscall_bls12381_fp2_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* Commutative */
        syscall_bls12381_fp2_addmod((unsigned long long *)ret, (const unsigned long long *)a);
    } else {
        vec_copy(ret, a, sizeof(vec384x));
        syscall_bls12381_fp2_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    }
#else
    add_mod_384x(ret, a, b, BLS12_381_P);
#endif
}"""

src = src.replace(old_add_fp2, new_add_fp2, 1)
print("Patched add_fp2")

# ── Direct inline replacement for sub_fp2 ──
old_sub_fp2 = """static inline void sub_fp2(vec384x ret, const vec384x a, const vec384x b)
{   sub_mod_384x(ret, a, b, BLS12_381_P);   }"""

new_sub_fp2 = """static inline void sub_fp2(vec384x ret, const vec384x a, const vec384x b)
{
#ifdef SP1_BLS12381_SYSCALLS
    if (ret == a) {
        syscall_bls12381_fp2_submod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* ret aliases b: need tmp because sub is non-commutative */
        vec384x tmp;
        vec_copy(tmp, a, sizeof(vec384x));
        syscall_bls12381_fp2_submod((unsigned long long *)tmp, (const unsigned long long *)b);
        vec_copy(ret, tmp, sizeof(vec384x));
    } else {
        /* ret is separate from both: copy a->ret, subtract in-place */
        vec_copy(ret, a, sizeof(vec384x));
        syscall_bls12381_fp2_submod((unsigned long long *)ret, (const unsigned long long *)b);
    }
#else
    sub_mod_384x(ret, a, b, BLS12_381_P);
#endif
}"""

src = src.replace(old_sub_fp2, new_sub_fp2, 1)
print("Patched sub_fp2")

# ── Direct inline replacement for sqr_fp2 ──
old_sqr_fp2 = """static inline void sqr_fp2(vec384x ret, const vec384x a)
{   sqr_mont_382x(ret, a, BLS12_381_P, p0);   }"""

new_sqr_fp2 = """static inline void sqr_fp2(vec384x ret, const vec384x a)
{
#ifdef SP1_BLS12381_SYSCALLS
    if ((const limb_t *)ret != (const limb_t *)a) vec_copy(ret, a, sizeof(vec384x));
    syscall_bls12381_fp2_mulmod((unsigned long long *)ret, (const unsigned long long *)ret);
    syscall_bls12381_fp2_mulmod((unsigned long long *)ret, _SP1_FP2_R_INV);
#else
    sqr_mont_382x(ret, a, BLS12_381_P, p0);
#endif
}"""

src = src.replace(old_sqr_fp2, new_sqr_fp2, 1)
print("Patched sqr_fp2")

with open(path, 'w') as f:
    f.write(src)

print("Patched " + path + " successfully")
PYEOF

# ────────── Phase 2: Patch no_asm.h (low-level Montgomery routines) ──────────
# With 64-bit limbs, all vec384/vec768/vec384x types are uint64_t arrays
# with native 8-byte alignment. No memcpy wrappers needed.
#
# OPTIMIZATIONS vs previous version:
#  - mul_mont_384: if ret==a, skip first vec_copy entirely
#  - sqr_mont_384: if ret==a, no copies needed at all
#  - from_mont_384: if ret==a, no copies needed
#  - add_mod_384 / sub_mod_384: patched with syscalls (catches mul_by_1_plus_i etc.)
python3 - "$NO_ASM_H" << 'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# ── Replace MUL_MONT_IMPL(384) with SP1-accelerated versions ──
# With 64-bit limbs, vec384 = uint64_t[6] is natively 8-byte aligned.
# We can pass pointers directly to syscalls.
# OPTIMIZATION: avoid vec_copy when ret==a (common case).

sp1_mul_sqr_384 = r"""/* SP1-patched: mul_mont_384 and sqr_mont_384 via syscalls (aliasing-aware) */
#ifdef SP1_BLS12381_SYSCALLS
static const unsigned long long _sp1_no_asm_R_INV[6] = {
    0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
    0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL
};

__attribute__((always_inline))
inline void mul_mont_384(vec384 ret, const vec384 a, const vec384 b,
                          const vec384 p, limb_t n0)
{
    (void)p; (void)n0;
    if (ret == a) {
        /* In-place: syscall writes directly to ret=a, zero copies */
        syscall_bls12381_fp_mulmod((unsigned long long *)ret, (const unsigned long long *)b);
        syscall_bls12381_fp_mulmod((unsigned long long *)ret, _sp1_no_asm_R_INV);
    } else if (ret == b) {
        /* ret aliases b: copy a->tmp, syscall(tmp, b), copy tmp->ret */
        vec384 tmp;
        vec_copy(tmp, a, sizeof(vec384));
        syscall_bls12381_fp_mulmod((unsigned long long *)tmp, (const unsigned long long *)b);
        syscall_bls12381_fp_mulmod((unsigned long long *)tmp, _sp1_no_asm_R_INV);
        vec_copy(ret, tmp, sizeof(vec384));
    } else {
        /* ret is separate: copy a->ret, syscall in-place, 1 copy */
        vec_copy(ret, a, sizeof(vec384));
        syscall_bls12381_fp_mulmod((unsigned long long *)ret, (const unsigned long long *)b);
        syscall_bls12381_fp_mulmod((unsigned long long *)ret, _sp1_no_asm_R_INV);
    }
}

__attribute__((always_inline))
inline void sqr_mont_384(vec384 ret, const vec384 a,
                          const vec384 p, limb_t n0)
{
    (void)p; (void)n0;
    /* sqr: almost always ret==a in blst. Zero copies in that case. */
    if (ret != a) vec_copy(ret, a, sizeof(vec384));
    syscall_bls12381_fp_mulmod((unsigned long long *)ret, (const unsigned long long *)ret);
    syscall_bls12381_fp_mulmod((unsigned long long *)ret, _sp1_no_asm_R_INV);
}
#else
MUL_MONT_IMPL(384)
#endif
"""

src = src.replace('MUL_MONT_IMPL(384)', sp1_mul_sqr_384, 1)

# ── Patch ADD_MOD_IMPL(384) to use syscalls ──
# This accelerates ALL 384-bit add_mod callers including:
# - mul_by_1_plus_i_mod_384x (used for mul_by_u_plus_1_fp2)
# - add_mod_384x (Fp2 add in vect.c)
# - sqr_mont_384x (in vect.c)
# All 384-bit operations in blst use BLS12-381 modulus, so this is safe.

old_add_384 = 'ADD_MOD_IMPL(384)'
new_add_384 = r"""/* SP1-patched: add_mod_384 via syscall (BLS12-381 only) */
#ifdef SP1_BLS12381_SYSCALLS
inline void add_mod_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p)
{
    (void)p;
    if (ret == a) {
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* Commutative: a+b = b+a */
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)a);
    } else {
        vec_copy(ret, a, sizeof(vec384));
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)b);
    }
}
#else
ADD_MOD_IMPL(384)
#endif"""

src = src.replace(old_add_384, new_add_384, 1)
print("Patched add_mod_384 with syscall")

# ── Patch SUB_MOD_IMPL(384) to use syscalls ──
old_sub_384 = 'SUB_MOD_IMPL(384)'
new_sub_384 = r"""/* SP1-patched: sub_mod_384 via syscall (BLS12-381 only) */
#ifdef SP1_BLS12381_SYSCALLS
inline void sub_mod_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p)
{
    (void)p;
    if (ret == a) {
        syscall_bls12381_fp_submod((unsigned long long *)ret, (const unsigned long long *)b);
    } else if (ret == b) {
        /* ret aliases b: non-commutative, need tmp */
        vec384 tmp;
        vec_copy(tmp, a, sizeof(vec384));
        syscall_bls12381_fp_submod((unsigned long long *)tmp, (const unsigned long long *)b);
        vec_copy(ret, tmp, sizeof(vec384));
    } else {
        /* ret is separate: copy a->ret, subtract in-place (1 copy) */
        vec_copy(ret, a, sizeof(vec384));
        syscall_bls12381_fp_submod((unsigned long long *)ret, (const unsigned long long *)b);
    }
}
#else
SUB_MOD_IMPL(384)
#endif"""

src = src.replace(old_sub_384, new_sub_384, 1)
print("Patched sub_mod_384 with syscall")

# ── Replace REDC_MONT_IMPL(384, 768) with SP1-accelerated version ──
# redc_mont_384(ret, a, p, n0): ret = a * R^{-1} mod P, where a is vec768.
# With 64-bit limbs: vec768 = uint64_t[12] (96 bytes, 8-byte aligned).
# Lower 384 bits = a[0..5], upper 384 bits = a[6..11].
# result = a_low * R^{-1} + a_high mod P

sp1_redc_384 = r"""/* SP1-patched: redc_mont_384 via syscalls (no aliasing possible) */
#ifdef SP1_BLS12381_SYSCALLS
__attribute__((always_inline))
inline void redc_mont_384(vec384 ret, const vec768 a,
                           const vec384 p, limb_t n0)
{
    (void)p; (void)n0;
    /* ret (vec384) cannot alias a (vec768) — different types/sizes.
       Copy a_low directly to ret, operate in-place. No tmp needed. */
    vec_copy(ret, a, sizeof(vec384));
    syscall_bls12381_fp_mulmod((unsigned long long *)ret, _sp1_no_asm_R_INV);
    syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)(a + NLIMBS(384)));
}
#else
REDC_MONT_IMPL(384, 768)
#endif
"""

src = src.replace('REDC_MONT_IMPL(384, 768)', sp1_redc_384, 1)

# ── Replace FROM_MONT_IMPL(384) with SP1-accelerated version ──
# from_mont_384(ret, a, p, n0) = a * R^{-1} mod P

sp1_from_384 = r"""/* SP1-patched: from_mont_384 via syscalls (aliasing-aware) */
#ifdef SP1_BLS12381_SYSCALLS
__attribute__((always_inline))
inline void from_mont_384(vec384 ret, const vec384 a,
                           const vec384 p, limb_t n0)
{
    (void)p; (void)n0;
    if (ret != a) vec_copy(ret, a, sizeof(vec384));
    syscall_bls12381_fp_mulmod((unsigned long long *)ret, _sp1_no_asm_R_INV);
}
#else
FROM_MONT_IMPL(384)
#endif
"""

src = src.replace('FROM_MONT_IMPL(384)', sp1_from_384, 1)

# ── Patch sqr_n_mul_mont_383 ──
# ret = a^{2^count} * b in Montgomery form.
# SP1 version: use our patched sqr_mont_384 + mul_mont_384.

old_sqr_n_mul = """void sqr_n_mul_mont_383(vec384 ret, const vec384 a, size_t count,
                        const vec384 p, limb_t n0, const vec384 b)
{
    __builtin_assume(count != 0);
    while(count--) {
        mul_mont_nonred_n(ret, a, a, p, n0, NLIMBS(384));
        a = ret;
    }
    mul_mont_n(ret, ret, b, p, n0, NLIMBS(384));
}"""

new_sqr_n_mul = """void sqr_n_mul_mont_383(vec384 ret, const vec384 a, size_t count,
                        const vec384 p, limb_t n0, const vec384 b)
{
#ifdef SP1_BLS12381_SYSCALLS
    __builtin_assume(count != 0);
    if (ret != a)
        vec_copy(ret, a, sizeof(vec384));
    /* sqr_mont_384(ret, ret, ...) will detect ret==a and skip copies */
    while(count--) {
        sqr_mont_384(ret, ret, p, n0);
    }
    mul_mont_384(ret, ret, b, p, n0);
#else
    __builtin_assume(count != 0);
    while(count--) {
        mul_mont_nonred_n(ret, a, a, p, n0, NLIMBS(384));
        a = ret;
    }
    mul_mont_n(ret, ret, b, p, n0, NLIMBS(384));
#endif
}"""

if old_sqr_n_mul in src:
    src = src.replace(old_sqr_n_mul, new_sqr_n_mul)
    print("Patched sqr_n_mul_mont_383 in no_asm.h")
else:
    print("WARNING: could not find sqr_n_mul_mont_383 in no_asm.h", file=sys.stderr)

# ── Patch mul_mont_384x (Fp2 Montgomery multiply) ──
# With 64-bit limbs, vec384x = uint64_t[12], natively 8-byte aligned.

old_mul_384x = """void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b,
                          const vec384 p, limb_t n0)
{
    vec384 aa, bb, cc;

    add_mod_n(aa, a[0], a[1], p, NLIMBS(384));
    add_mod_n(bb, b[0], b[1], p, NLIMBS(384));
    mul_mont_n(bb, bb, aa, p, n0, NLIMBS(384));
    mul_mont_n(aa, a[0], b[0], p, n0, NLIMBS(384));
    mul_mont_n(cc, a[1], b[1], p, n0, NLIMBS(384));
    sub_mod_n(ret[0], aa, cc, p, NLIMBS(384));
    sub_mod_n(ret[1], bb, aa, p, NLIMBS(384));
    sub_mod_n(ret[1], ret[1], cc, p, NLIMBS(384));
}"""

new_mul_384x = """void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b,
                          const vec384 p, limb_t n0)
{
#ifdef SP1_BLS12381_SYSCALLS
    (void)p; (void)n0;
    static const unsigned long long fp2_r_inv[12] = {
        0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
        0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL,
        0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
        0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL
    };
    if ((const limb_t *)ret == (const limb_t *)a) {
        /* In-place: zero copies */
        syscall_bls12381_fp2_mulmod((unsigned long long *)ret, (const unsigned long long *)b);
        syscall_bls12381_fp2_mulmod((unsigned long long *)ret, fp2_r_inv);
    } else if ((const limb_t *)ret == (const limb_t *)b) {
        /* ret aliases b: need tmp */
        vec384x tmp;
        vec_copy(tmp, a, sizeof(vec384x));
        syscall_bls12381_fp2_mulmod((unsigned long long *)tmp, (const unsigned long long *)b);
        syscall_bls12381_fp2_mulmod((unsigned long long *)tmp, fp2_r_inv);
        vec_copy(ret, tmp, sizeof(vec384x));
    } else {
        /* ret is separate: 1 copy */
        vec_copy(ret, a, sizeof(vec384x));
        syscall_bls12381_fp2_mulmod((unsigned long long *)ret, (const unsigned long long *)b);
        syscall_bls12381_fp2_mulmod((unsigned long long *)ret, fp2_r_inv);
    }
#else
    vec384 aa, bb, cc;

    add_mod_n(aa, a[0], a[1], p, NLIMBS(384));
    add_mod_n(bb, b[0], b[1], p, NLIMBS(384));
    mul_mont_n(bb, bb, aa, p, n0, NLIMBS(384));
    mul_mont_n(aa, a[0], b[0], p, n0, NLIMBS(384));
    mul_mont_n(cc, a[1], b[1], p, n0, NLIMBS(384));
    sub_mod_n(ret[0], aa, cc, p, NLIMBS(384));
    sub_mod_n(ret[1], bb, aa, p, NLIMBS(384));
    sub_mod_n(ret[1], ret[1], cc, p, NLIMBS(384));
#endif
}"""

if old_mul_384x in src:
    src = src.replace(old_mul_384x, new_mul_384x)
    print("Patched mul_mont_384x in no_asm.h")
else:
    print("WARNING: could not find mul_mont_384x in no_asm.h", file=sys.stderr)

# ── Patch sqr_mont_382x (used in fp12 tower for pairing) ──
# With 64-bit limbs, vec384x is natively 8-byte aligned.

old_sqr_382x = '''void sqr_mont_382x(vec384x ret, const vec384x a,
                          const vec384 p, limb_t n0)
{
    llimb_t limbx;
    limb_t mask, carry, borrow;
    size_t i;
    vec384 t0, t1;

    /* "add_mod_n(t0, a[0], a[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limbx = a[0][i] + (a[1][i] + (llimb_t)carry);
        t0[i] = (limb_t)limbx;
        carry = (limb_t)(limbx >> LIMB_T_BITS);
    }

    /* "sub_mod_n(t1, a[0], a[1], p, NLIMBS(384));" */
    for (borrow=0, i=0; i<NLIMBS(384); i++) {
        limbx = a[0][i] - (a[1][i] + (llimb_t)borrow);
        t1[i] = (limb_t)limbx;
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    }
    mask = 0 - borrow;
    launder(mask);

    /* "mul_mont_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));

    /* "add_mod_n(ret[1], ret[1], ret[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limb_t a_i = ret[1][i];
        ret[1][i] = a_i<<1 | carry;
        carry = a_i>>(LIMB_T_BITS-1);
    }

    /* "mul_mont_n(ret[0], t0, t1, p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[0], t0, t1, p, n0, NLIMBS(384));

    /* account for t1's sign... */
    for (borrow=0, i=0; i<NLIMBS(384); i++) {
        limbx = ret[0][i] - ((t0[i] & mask) + (llimb_t)borrow);
        ret[0][i] = (limb_t)limbx;
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    }
    mask = 0 - borrow;
    launder(mask);
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limbx = ret[0][i] + ((p[i] & mask) + (llimb_t)carry);
        ret[0][i] = (limb_t)limbx;
        carry = (limb_t)(limbx >> LIMB_T_BITS);
    }
}'''

new_sqr_382x = '''void sqr_mont_382x(vec384x ret, const vec384x a,
                          const vec384 p, limb_t n0)
{
#ifdef SP1_BLS12381_SYSCALLS
    (void)p; (void)n0;
    static const unsigned long long fp2_r_inv[12] = {
        0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
        0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL,
        0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
        0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL
    };
    /* sqr: almost always ret==a. Zero copies in that case. */
    if ((const limb_t *)ret != (const limb_t *)a) vec_copy(ret, a, sizeof(vec384x));
    syscall_bls12381_fp2_mulmod((unsigned long long *)ret, (const unsigned long long *)ret);
    syscall_bls12381_fp2_mulmod((unsigned long long *)ret, fp2_r_inv);
#else
    llimb_t limbx;
    limb_t mask, carry, borrow;
    size_t i;
    vec384 t0, t1;

    /* "add_mod_n(t0, a[0], a[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limbx = a[0][i] + (a[1][i] + (llimb_t)carry);
        t0[i] = (limb_t)limbx;
        carry = (limb_t)(limbx >> LIMB_T_BITS);
    }

    /* "sub_mod_n(t1, a[0], a[1], p, NLIMBS(384));" */
    for (borrow=0, i=0; i<NLIMBS(384); i++) {
        limbx = a[0][i] - (a[1][i] + (llimb_t)borrow);
        t1[i] = (limb_t)limbx;
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    }
    mask = 0 - borrow;
    launder(mask);

    /* "mul_mont_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));

    /* "add_mod_n(ret[1], ret[1], ret[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limb_t a_i = ret[1][i];
        ret[1][i] = a_i<<1 | carry;
        carry = a_i>>(LIMB_T_BITS-1);
    }

    /* "mul_mont_n(ret[0], t0, t1, p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[0], t0, t1, p, n0, NLIMBS(384));

    /* account for t1's sign... */
    for (borrow=0, i=0; i<NLIMBS(384); i++) {
        limbx = ret[0][i] - ((t0[i] & mask) + (llimb_t)borrow);
        ret[0][i] = (limb_t)limbx;
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    }
    mask = 0 - borrow;
    launder(mask);
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limbx = ret[0][i] + ((p[i] & mask) + (llimb_t)carry);
        ret[0][i] = (limb_t)limbx;
        carry = (limb_t)(limbx >> LIMB_T_BITS);
    }
#endif
}'''

if old_sqr_382x in src:
    src = src.replace(old_sqr_382x, new_sqr_382x)
    print("Patched sqr_mont_382x in no_asm.h")
else:
    print("WARNING: could not find sqr_mont_382x in no_asm.h", file=sys.stderr)

# ── Fix quot_rem_n: relax n%2==0 assumption ──
# With 64-bit limbs, NLIMBS(64)=1, but quot_rem_n has
# __builtin_assume(n != 0 && n%2 == 0) which compiles to
# if(!(n%2==0)) __builtin_unreachable(). With n=1 this is UB.
# The algorithm works fine with odd n; only the assumption is wrong.
old_quot_rem_assume = '    __builtin_assume(n != 0 && n%2 == 0);\n    llimb_t limbx;\n    limb_t tmp[n+1], carry, mask, borrow;\n    size_t i;\n\n    /* divisor*quotient */\n    for (carry=0, i=0; i<n; i++) {\n        limbx = (quotient * (llimb_t)divisor[i]) + carry;'
new_quot_rem_assume = '    __builtin_assume(n != 0);\n    llimb_t limbx;\n    limb_t tmp[n+1], carry, mask, borrow;\n    size_t i;\n\n    /* divisor*quotient */\n    for (carry=0, i=0; i<n; i++) {\n        limbx = (quotient * (llimb_t)divisor[i]) + carry;'

if old_quot_rem_assume in src:
    src = src.replace(old_quot_rem_assume, new_quot_rem_assume)
    print("Patched quot_rem_n: relaxed n%2==0 assumption in no_asm.h")
else:
    print("WARNING: could not find quot_rem_n assumption in no_asm.h", file=sys.stderr)

# ── Patch cneg_mod_384 with SP1 syscall ──
# cneg_mod_384(ret, a, flag, p) = flag ? (p - a) : a
# With SP1 syscalls: compute p - a via sub_mod syscall, then conditionally select.
# This is used in cneg_fp, cneg_fp2, neg_fp, neg_fp2, conjugate_fp12, frobenius_map.
# The C implementation does a loop over 6 limbs twice; the syscall does it in 1 ecall.
old_cneg_384 = 'CNEG_MOD_IMPL(384)'
new_cneg_384 = r"""/* SP1-patched: cneg_mod_384 via sub_mod syscall */
#ifdef SP1_BLS12381_SYSCALLS
inline void cneg_mod_384(vec384 ret, const vec384 a, bool_t flag, const vec384 p)
{
    (void)p;
    if (flag) {
        /* ret = -a mod p = 0 - a mod p */
        if (ret != a) {
            /* ret is separate: zero ret, subtract in-place (no tmp needed) */
            vec_zero(ret, sizeof(vec384));
            syscall_bls12381_fp_submod((unsigned long long *)ret, (const unsigned long long *)a);
        } else {
            /* ret aliases a: need tmp since zeroing ret would destroy a */
            vec384 z;
            vec_zero(z, sizeof(z));
            syscall_bls12381_fp_submod((unsigned long long *)z, (const unsigned long long *)a);
            vec_copy(ret, z, sizeof(vec384));
        }
    } else {
        if (ret != a) vec_copy(ret, a, sizeof(vec384));
    }
}
#else
CNEG_MOD_IMPL(384)
#endif"""

if old_cneg_384 in src:
    src = src.replace(old_cneg_384, new_cneg_384)
    print("Patched cneg_mod_384 with SP1 syscall")
else:
    print("WARNING: could not find CNEG_MOD_IMPL(384) in no_asm.h", file=sys.stderr)

# ── Patch lshift_mod_384 to use add_mod_384 syscalls ──
# The no_asm.h version uses generic C loops. Replace with repeated add_mod_384
# (which is already syscall-accelerated). Left-shift by 1 = add(a, a).
# This catches lshift_fp, lshift_fp2, mul_by_8_fp, mul_by_8_fp2.
old_lshift_384 = 'LSHIFT_MOD_IMPL(384)'
new_lshift_384 = r"""/* SP1-patched: lshift_mod_384 via repeated add_mod_384 syscall */
#ifdef SP1_BLS12381_SYSCALLS
inline void lshift_mod_384(vec384 ret, const vec384 a, size_t count, const vec384 p)
{
    (void)p;
    if (ret != a) vec_copy(ret, a, sizeof(vec384));
    while (count--) {
        syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)ret);
    }
}
#else
LSHIFT_MOD_IMPL(384)
#endif"""

if old_lshift_384 in src:
    src = src.replace(old_lshift_384, new_lshift_384)
    print("Patched lshift_mod_384 with SP1 add syscall")
else:
    print("WARNING: could not find LSHIFT_MOD_IMPL(384) in no_asm.h", file=sys.stderr)

# ── Patch mul_by_3_mod_384 to use add_mod syscalls directly ──
# mul_by_3(a) = a + a + a = 2 add syscalls.
# The C version does 6 limb shifts + 2*6 limb sub/compare = ~36 ops.
# With syscalls: 2 ecalls (add self, then add original).
old_mul_by_3_384 = 'MUL_BY_3_MOD_IMPL(384)'
new_mul_by_3_384 = r"""/* SP1-patched: mul_by_3_mod_384 via add_mod_384 syscall */
#ifdef SP1_BLS12381_SYSCALLS
inline void mul_by_3_mod_384(vec384 ret, const vec384 a, const vec384 p)
{
    (void)p;
    /* ret = a + a = 2a */
    if (ret != a) vec_copy(ret, a, sizeof(vec384));
    vec384 orig;
    vec_copy(orig, a, sizeof(vec384));
    syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)ret);
    /* ret = 2a + a = 3a */
    syscall_bls12381_fp_addmod((unsigned long long *)ret, (const unsigned long long *)orig);
}
#else
MUL_BY_3_MOD_IMPL(384)
#endif"""

if old_mul_by_3_384 in src:
    src = src.replace(old_mul_by_3_384, new_mul_by_3_384)
    print("Patched mul_by_3_mod_384 with SP1 add syscall")
else:
    print("WARNING: could not find MUL_BY_3_MOD_IMPL(384) in no_asm.h", file=sys.stderr)

with open(path, 'w') as f:
    f.write(src)

print("Patched " + path + " successfully")
PYEOF

echo "patch_blst_sp1.sh: done"
