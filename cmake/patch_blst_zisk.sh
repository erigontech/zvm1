#!/bin/sh
# Copyright 2026 The zvm1 Authors (modifications)
# Patch blst Fp/Fp2 arithmetic onto ZisK precompiles.
set -e

VECT_H="src/vect.h"
FIELDS_H="src/fields.h"
NO_ASM_H="src/no_asm.h"
VECT_C="src/vect.c"
RECIP_C="src/recip.c"
for f in "$VECT_H" "$FIELDS_H" "$NO_ASM_H" "$VECT_C" "$RECIP_C"; do
    [ -f "$f" ] || { echo "ERR: $f not found (pwd=$(pwd))"; exit 1; }
done

grep -q "ZISK_BLS12381_SYSCALLS" "$VECT_H" && { echo "Already patched"; exit 0; }

python3 - "$VECT_H" "$FIELDS_H" "$NO_ASM_H" "$VECT_C" "$RECIP_C" << 'PYEOF'
import sys

VECT_H, FIELDS_H, NO_ASM_H, VECT_C, RECIP_C = sys.argv[1:6]


def patch(path, edits):
    with open(path) as f:
        src = f.read()
    for name, old, new in edits:
        if src.count(old) != 1:
            sys.exit("ERR: %s: expected one match for %s, found %d" % (path, name, src.count(old)))
        src = src.replace(old, new, 1)
        print("Patched %s: %s" % (path, name))
    with open(path, "w") as f:
        f.write(src)


def gated(new, old):
    return "#ifdef ZISK_BLS12381_SYSCALLS\n" + new + "#else\n" + old + "\n#endif"


# ────────── Phase 0: 64-bit limbs ──────────
old_limb = """#elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
typedef unsigned int limb_t;
# define LIMB_T_BITS    32
# ifndef __BLST_NO_ASM__
#  define __BLST_NO_ASM__
# endif"""

new_limb = """#elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
# if defined(__riscv) && (__riscv_xlen == 64)
typedef unsigned long long limb_t;
#  define LIMB_T_BITS    64
# else
typedef unsigned int limb_t;
#  define LIMB_T_BITS    32
# endif
# ifndef __BLST_NO_ASM__
#  define __BLST_NO_ASM__
# endif"""

# ────────── Phase 1: CSR helpers (end of vect.h) ──────────
helpers = r"""
#ifdef ZISK_BLS12381_SYSCALLS
# if LIMB_T_BITS != 64
#  error "ZisK precompiles need 64-bit limbs"
# endif

void zisk_abort(void) __attribute__((noreturn));

typedef struct { const limb_t *a, *b, *c, *m; limb_t *d; } zisk_arith384_t;
typedef struct { limb_t *f1; const limb_t *f2; } zisk_fp2_t;

static const limb_t zisk_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
static const limb_t zisk_Pm1[6] = {
    0xb9feffffffffaaaaULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
/* (2^384)^-1 mod P; as Fp2 (R^-1, 0) */
static const limb_t zisk_Rinv[12] = {
    0xf4d38259380b4820ULL, 0x7fe11274d898fafbULL, 0x343ea97914956dc8ULL,
    0x1797ab1458a88de9ULL, 0xed5e64273c4f538bULL, 0x14fec701e8fb0ce9ULL,
    0, 0, 0, 0, 0, 0
};
/* (2^384)^2 mod P */
static const limb_t zisk_RR[6] = {
    0xf4df1f341c341746ULL, 0x0a76e6a609d104f1ULL, 0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL, 0x9a793e85b519952dULL, 0x11988fe592cae3aaULL
};
/* 2^-1 mod P */
static const limb_t zisk_inv2[6] = {
    0xdcff7fffffffd556ULL, 0x0f55ffff58a9ffffULL, 0xb39869507b587b12ULL,
    0xb23ba5c279c2895fULL, 0x258dd3db21a5d66bULL, 0x0d0088f51cbff34dULL
};
/* 1; as Fp2 1 + i */
static const limb_t zisk_one[12] = { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
static const limb_t zisk_three[6] = { 3, 0, 0, 0, 0, 0 };
static const limb_t zisk_zero[6] = { 0, 0, 0, 0, 0, 0 };

/* Reads precede writes: outputs may alias. */
static inline __attribute__((always_inline))
void zisk_arith384(limb_t *d, const limb_t *a, const limb_t *b, const limb_t *c)
{
    const zisk_arith384_t prm = { a, b, c, zisk_P, d };
    __asm__ volatile("csrs 0x80B, %0" : : "r"(&prm) : "memory");
}

#define ZISK_FP2_OP(name, csr)                                              \
static inline __attribute__((always_inline))                                \
void name(limb_t *f1, const limb_t *f2)                                     \
{                                                                           \
    const zisk_fp2_t prm = { f1, f2 };                                      \
    __asm__ volatile("csrs " #csr ", %0" : : "r"(&prm) : "memory");         \
}
ZISK_FP2_OP(zisk_fp2_add, 0x80E)
ZISK_FP2_OP(zisk_fp2_sub, 0x80F)
ZISK_FP2_OP(zisk_fp2_mul, 0x810)
#undef ZISK_FP2_OP

static inline __attribute__((always_inline))
void zisk_add_mod(limb_t *ret, const limb_t *a, const limb_t *b)
{   zisk_arith384(ret, a, zisk_one, b);   }

static inline __attribute__((always_inline))
void zisk_sub_mod(limb_t *ret, const limb_t *a, const limb_t *b)
{   zisk_arith384(ret, b, zisk_Pm1, a);   }

static inline __attribute__((always_inline))
void zisk_mul_mont(limb_t *ret, const limb_t *a, const limb_t *b)
{
    zisk_arith384(ret, a, b, zisk_zero);
    zisk_arith384(ret, ret, zisk_Rinv, zisk_zero);
}

/* ret, a, b are equal or disjoint */
static inline __attribute__((always_inline))
void zisk_fp2_mul_mont(limb_t *ret, const limb_t *a, const limb_t *b)
{
    if (ret == b)
        b = a;
    else if (ret != a)
        vec_copy(ret, a, 12 * sizeof(limb_t));
    zisk_fp2_mul(ret, b);
    zisk_fp2_mul(ret, zisk_Rinv);
}

static inline __attribute__((always_inline))
void zisk_fp2_add_mod(limb_t *ret, const limb_t *a, const limb_t *b)
{
    if (ret == a)
        zisk_fp2_add(ret, b);
    else if (ret == b)
        zisk_fp2_add(ret, a);
    else {
        zisk_add_mod(ret, a, b);
        zisk_add_mod(ret + 6, a + 6, b + 6);
    }
}

static inline __attribute__((always_inline))
void zisk_fp2_sub_mod(limb_t *ret, const limb_t *a, const limb_t *b)
{
    if (ret == a)
        zisk_fp2_sub(ret, b);
    else {
        zisk_sub_mod(ret, a, b);
        zisk_sub_mod(ret + 6, a + 6, b + 6);
    }
}

static inline int zisk_lt_P(const limb_t *a)
{
    size_t i;
    for (i = 6; i--;)
        if (a[i] != zisk_P[i])
            return a[i] < zisk_P[i];
    return 0;
}

/* fcall 10; port 0x8F3 reads 8 words */
static inline __attribute__((always_inline))
void zisk_fcall_fp_inv(limb_t y[6], const limb_t x[6])
{
    const limb_t in[8] = { x[0], x[1], x[2], x[3], x[4], x[5], 0, 0 };
    __asm__ volatile(
        "csrs 0x8F3, %6\n\t"
        "csrwi 0x8C0, 10\n\t"
        "csrr %0, 0xFFE\n\t"
        "csrr %1, 0xFFE\n\t"
        "csrr %2, 0xFFE\n\t"
        "csrr %3, 0xFFE\n\t"
        "csrr %4, 0xFFE\n\t"
        "csrr %5, 0xFFE"
        : "=r"(y[0]), "=r"(y[1]), "=r"(y[2]), "=r"(y[3]), "=r"(y[4]), "=r"(y[5])
        : "r"(in)
        : "memory");
}

/* Untrusted hint: x canonical and nonzero */
static void zisk_check_fp_inv(const limb_t x[6], const limb_t y[6])
{
    limb_t t[6];

    zisk_arith384(t, x, y, zisk_zero);
    if (!zisk_lt_P(y) || !vec_is_equal(t, zisk_one, sizeof(t)))
        zisk_abort();
}

/* Montgomery 1/inp; 0 maps to 0 */
static void zisk_reciprocal_fp(limb_t out[6], const limb_t inp[6])
{
    limb_t x[6], y[6];

    zisk_arith384(x, inp, zisk_one, zisk_zero);
    if (vec_is_zero(x, sizeof(x))) {
        vec_zero(out, sizeof(x));
        return;
    }
    zisk_fcall_fp_inv(y, x);
    zisk_check_fp_inv(x, y);
    zisk_arith384(out, y, zisk_RR, zisk_zero);
}
#endif /* ZISK_BLS12381_SYSCALLS */

#endif /* __BLS12_381_ASM_VECT_H__ */"""

# ZisK DMA mem ops beat limb loops.
old_vec_copy = """    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
}"""

new_vec_copy = """#ifdef ZISK_BLS12381_SYSCALLS
    (void)rp; (void)ap; (void)i;
    memcpy(ret, a, num);
#else
    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
#endif
}"""

old_vec_zero = """    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = 0;
"""

new_vec_zero = """#ifdef ZISK_BLS12381_SYSCALLS
    (void)rp; (void)i;
    memset(ret, 0, num);
#else
    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = 0;
#endif
"""

old_vec_select = """#else
    if (0) ;
#endif
    else {"""

new_vec_select = """#elif defined(ZISK_BLS12381_SYSCALLS)
    if (1)
        memmove(ret, sel_a ? a : b, num);
#else
    if (0) ;
#endif
    else {"""

old_vec_is_equal = """#ifndef __BLST_NO_ASM__
    bool_t vec_is_equal_16x(const void *a, const void *b, size_t num);"""

new_vec_is_equal = """#ifdef ZISK_BLS12381_SYSCALLS
    (void)ap; (void)bp; (void)acc; (void)i;
    return memcmp(a, b, num) == 0;
#endif
""" + old_vec_is_equal

patch(VECT_H, [
    ("64-bit limbs on rv64", old_limb, new_limb),
    ("<string.h>", "#include <stddef.h>\n",
     "#include <stddef.h>\n#ifdef ZISK_BLS12381_SYSCALLS\n# include <string.h>\n#endif\n"),
    ("vec_select", old_vec_select, new_vec_select),
    ("vec_is_equal", old_vec_is_equal, new_vec_is_equal),
    ("vec_copy", old_vec_copy, new_vec_copy),
    ("vec_zero", old_vec_zero, new_vec_zero),
    ("ZisK CSR helpers", "\n#endif /* __BLS12_381_ASM_VECT_H__ */", helpers),
])

# ────────── Phase 2: fields.h Fp2 shortcuts, inlined ──────────
def fp2_shortcut(name, call, zisk):
    old = "static inline void %s{   %s;   }" % (name, call)
    new = "static inline void %s{\n#ifdef ZISK_BLS12381_SYSCALLS\n    %s;\n#else\n    %s;\n#endif\n}" % (name, zisk, call)
    return ("%s" % name.split("(")[0], old, new)

patch(FIELDS_H, [
    fp2_shortcut("add_fp2(vec384x ret, const vec384x a, const vec384x b)\n",
                 "add_mod_384x(ret, a, b, BLS12_381_P)",
                 "zisk_fp2_add_mod((limb_t *)ret, (const limb_t *)a, (const limb_t *)b)"),
    fp2_shortcut("sub_fp2(vec384x ret, const vec384x a, const vec384x b)\n",
                 "sub_mod_384x(ret, a, b, BLS12_381_P)",
                 "zisk_fp2_sub_mod((limb_t *)ret, (const limb_t *)a, (const limb_t *)b)"),
    fp2_shortcut("mul_fp2(vec384x ret, const vec384x a, const vec384x b)\n",
                 "mul_mont_384x(ret, a, b, BLS12_381_P, p0)",
                 "zisk_fp2_mul_mont((limb_t *)ret, (const limb_t *)a, (const limb_t *)b)"),
    fp2_shortcut("sqr_fp2(vec384x ret, const vec384x a)\n",
                 "sqr_mont_384x(ret, a, BLS12_381_P, p0)",
                 "zisk_fp2_mul_mont((limb_t *)ret, (const limb_t *)a, (const limb_t *)a)"),
])

# ────────── Phase 0 + 2: no_asm.h ──────────
old_llimb = """#if LIMB_T_BITS==32
typedef unsigned long long llimb_t;
#endif"""

new_llimb = """#if LIMB_T_BITS==32
typedef unsigned long long llimb_t;
#elif LIMB_T_BITS==64
typedef unsigned __int128 llimb_t;
#endif"""

# NLIMBS(64) == 1 breaks the n%2 assumption.
old_quot_rem = """static limb_t quot_rem_n(limb_t *div_rem, const limb_t *divisor,
                                          limb_t quotient, size_t n)
{
    __builtin_assume(n != 0 && n%2 == 0);"""

new_quot_rem = """static limb_t quot_rem_n(limb_t *div_rem, const limb_t *divisor,
                                          limb_t quotient, size_t n)
{
    __builtin_assume(n != 0);"""

mul_sqr_384 = """inline void mul_mont_384(vec384 ret, const vec384 a, const vec384 b,
                         const vec384 p, limb_t n0)
{   (void)p; (void)n0; zisk_mul_mont(ret, a, b);   }

inline void sqr_mont_384(vec384 ret, const vec384 a,
                         const vec384 p, limb_t n0)
{   (void)p; (void)n0; zisk_mul_mont(ret, a, a);   }
"""

add_384 = """inline void add_mod_384(vec384 ret, const vec384 a, const vec384 b,
                        const vec384 p)
{   (void)p; zisk_add_mod(ret, a, b);   }
"""

sub_384 = """inline void sub_mod_384(vec384 ret, const vec384 a, const vec384 b,
                        const vec384 p)
{   (void)p; zisk_sub_mod(ret, a, b);   }
"""

mul_by_3_384 = """inline void mul_by_3_mod_384(vec384 ret, const vec384 a, const vec384 p)
{   (void)p; zisk_arith384(ret, a, zisk_three, zisk_zero);   }
"""

lshift_384 = """inline void lshift_mod_384(vec384 ret, const vec384 a, size_t count,
                           const vec384 p)
{
    limb_t k[6] = { 0, 0, 0, 0, 0, 0 };
    size_t s;

    (void)p;
    while (count) {
        s = count < 63 ? count : 63;
        k[0] = (limb_t)1 << s;
        zisk_arith384(ret, a, k, zisk_zero);
        a = ret;
        count -= s;
    }
}
"""

cneg_384 = """inline void cneg_mod_384(vec384 ret, const vec384 a, bool_t flag,
                         const vec384 p)
{
    (void)p;
    if (flag & 1)
        zisk_arith384(ret, a, zisk_Pm1, zisk_zero);
    else if (ret != a)
        vec_copy(ret, a, sizeof(vec384));
}
"""

from_384 = """inline void from_mont_384(vec384 ret, const vec384 a,
                         const vec384 p, limb_t n0)
{   (void)p; (void)n0; zisk_arith384(ret, a, zisk_Rinv, zisk_zero);   }
"""

# a * R^-1 = a_lo * R^-1 + a_hi
redc_384 = """inline void redc_mont_384(vec384 ret, const vec768 a,
                         const vec384 p, limb_t n0)
{   (void)p; (void)n0; zisk_arith384(ret, a, zisk_Rinv, a + NLIMBS(384));   }
"""

rshift_384 = """inline void rshift_mod_384(vec384 ret, const vec384 a, size_t count,
                           const vec384 p)
{
    (void)p;
    while (count--) {
        zisk_arith384(ret, a, zisk_inv2, zisk_zero);
        a = ret;
    }
}
"""

div_by_2_384 = """inline void div_by_2_mod_384(vec384 ret, const vec384 a,
                            const vec384 p)
{   (void)p; zisk_arith384(ret, a, zisk_inv2, zisk_zero);   }
"""

old_mul_384x_body = """{
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

new_mul_384x_body = """{
#ifdef ZISK_BLS12381_SYSCALLS
    (void)p; (void)n0;
    zisk_fp2_mul_mont((limb_t *)ret, (const limb_t *)a, (const limb_t *)b);
#else
""" + old_mul_384x_body[2:-2] + """
#endif
}"""

old_sqr_n_mul = """{
    __builtin_assume(count != 0);
    while(count--) {
        mul_mont_nonred_n(ret, a, a, p, n0, NLIMBS(384));
        a = ret;
    }
    mul_mont_n(ret, ret, b, p, n0, NLIMBS(384));
}"""

new_sqr_n_mul = """{
#ifdef ZISK_BLS12381_SYSCALLS
    (void)p; (void)n0;
    while(count--) {
        zisk_mul_mont(ret, a, a);
        a = ret;
    }
    zisk_mul_mont(ret, ret, b);
#else
""" + old_sqr_n_mul[2:-2] + """
#endif
}"""

patch(NO_ASM_H, [
    ("__int128 llimb_t", old_llimb, new_llimb),
    ("quot_rem_n n%2 assumption", old_quot_rem, new_quot_rem),
    ("mul/sqr_mont_384", "MUL_MONT_IMPL(384)", gated(mul_sqr_384, "MUL_MONT_IMPL(384)")),
    ("add_mod_384", "ADD_MOD_IMPL(384)", gated(add_384, "ADD_MOD_IMPL(384)")),
    ("sub_mod_384", "SUB_MOD_IMPL(384)", gated(sub_384, "SUB_MOD_IMPL(384)")),
    ("mul_by_3_mod_384", "MUL_BY_3_MOD_IMPL(384)", gated(mul_by_3_384, "MUL_BY_3_MOD_IMPL(384)")),
    ("lshift_mod_384", "LSHIFT_MOD_IMPL(384)", gated(lshift_384, "LSHIFT_MOD_IMPL(384)")),
    ("cneg_mod_384", "CNEG_MOD_IMPL(384)", gated(cneg_384, "CNEG_MOD_IMPL(384)")),
    ("from_mont_384", "FROM_MONT_IMPL(384)", gated(from_384, "FROM_MONT_IMPL(384)")),
    ("redc_mont_384", "REDC_MONT_IMPL(384, 768)", gated(redc_384, "REDC_MONT_IMPL(384, 768)")),
    ("rshift_mod_384", "RSHIFT_MOD_IMPL(384)", gated(rshift_384, "RSHIFT_MOD_IMPL(384)")),
    ("div_by_2_mod_384", "DIV_BY_2_MOD_IMPL(384)", gated(div_by_2_384, "DIV_BY_2_MOD_IMPL(384)")),
    ("mul_mont_384x", old_mul_384x_body, new_mul_384x_body),
    ("sqr_n_mul_mont_383", old_sqr_n_mul, new_sqr_n_mul),
    ("sgn0_pty_mont_384",
     "    from_mont_n(tmp, a, p, n0, NLIMBS(384));\n",
     "    from_mont_384(tmp, a, p, n0);\n"),
    ("sgn0_pty_mont_384x",
     "    from_mont_n(tmp[0], a[0], p, n0, NLIMBS(384));\n"
     "    from_mont_n(tmp[1], a[1], p, n0, NLIMBS(384));\n",
     "    from_mont_384(tmp[0], a[0], p, n0);\n"
     "    from_mont_384(tmp[1], a[1], p, n0);\n"),
])

# ────────── Phase 2: vect.c Fp2 reference routines ──────────
def body_gate(old, new):
    return old, "{\n#ifdef ZISK_BLS12381_SYSCALLS\n" + new + "#else\n" + old[2:-2] + "\n#endif\n}"

old_mul_by_1_plus_i = """{
    vec384 t;

    add_mod_384(t, a[0], a[1], mod);
    sub_mod_384(ret[0], a[0], a[1], mod);
    vec_copy(ret[1], t, sizeof(t));
}"""
new_mul_by_1_plus_i = """    (void)mod;
    if (ret == a) {
        zisk_fp2_mul((limb_t *)ret, zisk_one);
    } else {
        zisk_sub_mod(ret[0], a[0], a[1]);
        zisk_add_mod(ret[1], a[0], a[1]);
    }
"""

old_add_384x = """{
    add_mod_384(ret[0], a[0], b[0], mod);
    add_mod_384(ret[1], a[1], b[1], mod);
}"""
new_add_384x = """    (void)mod;
    zisk_fp2_add_mod((limb_t *)ret, (const limb_t *)a, (const limb_t *)b);
"""

old_sub_384x = """{
    sub_mod_384(ret[0], a[0], b[0], mod);
    sub_mod_384(ret[1], a[1], b[1], mod);
}"""
new_sub_384x = """    (void)mod;
    zisk_fp2_sub_mod((limb_t *)ret, (const limb_t *)a, (const limb_t *)b);
"""

old_sqr_384x = """{
    vec384 t0, t1;

    add_mod_384(t0, a[0], a[1], mod);
    sub_mod_384(t1, a[0], a[1], mod);

    mul_mont_384(ret[1], a[0], a[1], mod, n0);
    add_mod_384(ret[1], ret[1], ret[1], mod);

    mul_mont_384(ret[0], t0, t1, mod, n0);
}"""
new_sqr_384x = """    (void)mod; (void)n0;
    zisk_fp2_mul_mont((limb_t *)ret, (const limb_t *)a, (const limb_t *)a);
"""

patch(VECT_C, [
    ("mul_by_1_plus_i_mod_384x", *body_gate(old_mul_by_1_plus_i, new_mul_by_1_plus_i)),
    ("add_mod_384x", *body_gate(old_add_384x, new_add_384x)),
    ("sub_mod_384x", *body_gate(old_sub_384x, new_sub_384x)),
    ("sqr_mont_384x", *body_gate(old_sqr_384x, new_sqr_384x)),
])

# ────────── Phase 3: recip.c ──────────
old_recip = """    union { vec768 x; vec384 r[2]; } temp;

    ct_inverse_mod_384(temp.x, inp, BLS12_381_P, Px8);"""
new_recip = """    union { vec768 x; vec384 r[2]; } temp;

#ifdef ZISK_BLS12381_SYSCALLS
    (void)Px8; (void)temp;
    zisk_reciprocal_fp(out, inp);
    return;
#endif
    ct_inverse_mod_384(temp.x, inp, BLS12_381_P, Px8);"""

patch(RECIP_C, [("reciprocal_fp", old_recip, new_recip)])
PYEOF

echo "patch_blst_zisk.sh: done"
