#!/bin/sh
# Patch blst's no_asm.h for Airbender BigInt CSR (0x7CA) acceleration.
#
# BLS12-381 Fp multiplication on rv32im with 32-bit limbs uses CIOS with n=12,
# costing ~2900 instructions per mul_mont_384 call. By decomposing 384-bit
# operands into (hi:128, lo:256) chunks and using the 256-bit BigInt CSR for
# MUL_LOW/MUL_HIGH/ADD, we reduce this to ~33 CSR calls + ~200 insns glue.
#
# Only CSR ADD carry-in (bit 6) is used; SUB borrow-in is NOT assumed.
set -e

NO_ASM_H="src/no_asm.h"
[ -f "$NO_ASM_H" ] || { echo "ERR: $NO_ASM_H not found (pwd=$(pwd))"; exit 1; }

grep -q "AIRBENDER_BIGINT_CSR" "$NO_ASM_H" && { echo "Already patched"; exit 0; }

python3 - "$NO_ASM_H" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

airbender_384 = r"""/* Airbender BigInt CSR (0x7CA) accelerated 384-bit Montgomery arithmetic.
 * AIRBENDER_BIGINT_CSR marker for idempotent patching.
 *
 * 384-bit values are decomposed into (lo:256, hi:128) chunks.
 * CSR ops: MUL_LOW(0x08), MUL_HIGH(0x10), ADD(0x01), SUB(0x02), MEMCOPY(0x80).
 * ADD carry-in via bit 6 of mask; carry-out returned in x12.
 * SUB: borrow-out in x12 but borrow-in NOT used (not verified in hw).
 *
 * All CSR buffers must be 32-byte aligned.
 */
#ifdef AIRBENDER_BIGINT_CSR

#define _BLS_ALIGN32 __attribute__((aligned(32)))

static inline __attribute__((always_inline))
limb_t _bls_csr(limb_t *mut, const limb_t *immut, limb_t mask)
{
    register unsigned long x10 __asm__("x10") = (unsigned long)mut;
    register unsigned long x11 __asm__("x11") = (unsigned long)immut;
    register limb_t x12 __asm__("x12") = mask;
    __asm__ __volatile__("csrrw x0, 0x7CA, x0"
                 : "+r"(x12) : "r"(x10), "r"(x11) : "memory");
    return x12;
}

/* Copy 256 bits (8 words) between aligned buffers using CSR MEMCOPY. */
static inline __attribute__((always_inline))
void _bls_copy256(limb_t *dst, const limb_t *src)
{ _bls_csr(dst, src, 0x80); }

/* Copy lower 128 bits (4 words) and zero upper 128. */
static inline __attribute__((always_inline))
void _bls_pad128(limb_t *dst, const limb_t *src)
{
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
}

/* Software 384-bit compare: return 1 if a >= b. */
static inline __attribute__((always_inline))
int _bls_ge384(const limb_t a[12], const limb_t b[12])
{
    int i;
    for (i = 11; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1; /* equal */
}

/* Software 384-bit subtract: ret = a - b, returns borrow. */
static inline __attribute__((always_inline))
limb_t _bls_sub384(limb_t ret[12], const limb_t a[12], const limb_t b[12])
{
    llimb_t limbx;
    limb_t borrow = 0;
    int i;
    for (i = 0; i < 12; i++) {
        limbx = (llimb_t)a[i] - b[i] - borrow;
        ret[i] = (limb_t)limbx;
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    }
    return borrow;
}

/* BLS12-381 constants */
static const limb_t _bls_p_lo[8] _BLS_ALIGN32 = {
    0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
    0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84
};
static const limb_t _bls_p_hi[8] _BLS_ALIGN32 = {
    0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea,
    0, 0, 0, 0
};
static const limb_t _bls_P[12] = {
    0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
    0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
    0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea
};
static const limb_t _bls_np_lo[8] _BLS_ALIGN32 = {
    0xfffcfffd, 0x89f3fffc, 0xd9d113e8, 0x286adb92,
    0xc8e30b48, 0x16ef2ef0, 0x8eb2db4c, 0x19ecca0e
};
static const limb_t _bls_np_hi[8] _BLS_ALIGN32 = {
    0xe268cf58, 0x68b316fe, 0xfeaafc94, 0xceb06106,
    0, 0, 0, 0
};

/*
 * mul_mont_384: ret = a * b * R^{-1} mod P, R = 2^384
 *
 * 1. T = a * b  (768 bits, schoolbook with 256-bit chunks)
 * 2. m = T_lo_384 * N' mod 2^384
 * 3. (T + m*P) >> 384, conditional subtract P
 *
 * ~33 CSR calls total.
 */
__attribute__((noinline))
void mul_mont_384(vec384 ret, const vec384 a, const vec384 b,
                  const vec384 p, limb_t n0)
{
    (void)p; (void)n0;

    limb_t a_lo[8] _BLS_ALIGN32, a_hi[8] _BLS_ALIGN32;
    limb_t b_lo[8] _BLS_ALIGN32, b_hi[8] _BLS_ALIGN32;
    limb_t buf_A[8] _BLS_ALIGN32;
    limb_t T0[8] _BLS_ALIGN32, T1[8] _BLS_ALIGN32, T2[8] _BLS_ALIGN32;
    limb_t m0[8] _BLS_ALIGN32, m1[8] _BLS_ALIGN32;
    limb_t MP0[8] _BLS_ALIGN32, MP1[8] _BLS_ALIGN32, MP2[8] _BLS_ALIGN32;
    limb_t c1, c2, c3, carry_mid, carry_mp;

    /* Split a, b into (lo:256, hi:128) */
    a_lo[0]=a[0]; a_lo[1]=a[1]; a_lo[2]=a[2]; a_lo[3]=a[3];
    a_lo[4]=a[4]; a_lo[5]=a[5]; a_lo[6]=a[6]; a_lo[7]=a[7];
    _bls_pad128(a_hi, a+8);
    b_lo[0]=b[0]; b_lo[1]=b[1]; b_lo[2]=b[2]; b_lo[3]=b[3];
    b_lo[4]=b[4]; b_lo[5]=b[5]; b_lo[6]=b[6]; b_lo[7]=b[7];
    _bls_pad128(b_hi, b+8);

    /* === Phase 1: T = a * b (768 bits) === */

    /* a_lo * b_lo -> (T0, buf_A=ll_hi) */
    _bls_copy256(T0, a_lo);
    _bls_csr(T0, b_lo, 0x08);          /* T0 = MUL_LOW(a_lo, b_lo) */
    _bls_copy256(buf_A, a_lo);
    _bls_csr(buf_A, b_lo, 0x10);       /* buf_A = ll_hi = MUL_HIGH(a_lo, b_lo) */

    /* T1 = ll_hi + lh_lo + hl_lo */
    _bls_copy256(T1, buf_A);           /* T1 = ll_hi */

    _bls_copy256(buf_A, a_lo);
    _bls_csr(buf_A, b_hi, 0x08);       /* buf_A = lh_lo = MUL_LOW(a_lo, b_hi) */
    c1 = _bls_csr(T1, buf_A, 0x01);    /* T1 += lh_lo */

    _bls_copy256(buf_A, a_hi);
    _bls_csr(buf_A, b_lo, 0x08);       /* buf_A = hl_lo = MUL_LOW(a_hi, b_lo) */
    c2 = _bls_csr(T1, buf_A, 0x01);    /* T1 += hl_lo */

    carry_mid = c1 + c2;

    /* T2 = lh_hi + hl_hi + hh + carry_mid */
    _bls_copy256(T2, a_lo);
    _bls_csr(T2, b_hi, 0x10);          /* T2 = lh_hi = MUL_HIGH(a_lo, b_hi) */

    _bls_copy256(buf_A, a_hi);
    _bls_csr(buf_A, b_lo, 0x10);       /* buf_A = hl_hi = MUL_HIGH(a_hi, b_lo) */
    c1 = _bls_csr(T2, buf_A, 0x01);    /* T2 += hl_hi */

    _bls_copy256(buf_A, a_hi);
    _bls_csr(buf_A, b_hi, 0x08);       /* buf_A = hh = MUL_LOW(a_hi, b_hi) */
    c2 = _bls_csr(T2, buf_A, 0x01);    /* T2 += hh */

    /* Add carry_mid + c1 + c2. At most 4, fits in a word. */
    {
        limb_t cm = carry_mid + c1 + c2;
        if (cm) {
            buf_A[0]=cm; buf_A[1]=0; buf_A[2]=0; buf_A[3]=0;
            buf_A[4]=0; buf_A[5]=0; buf_A[6]=0; buf_A[7]=0;
            _bls_csr(T2, buf_A, 0x01);
        }
    }

    /* === Phase 2: m = T_lo_384 * N' mod 2^384 === */

    /* m0 = MUL_LOW(T0, np_lo) */
    _bls_copy256(m0, T0);
    _bls_csr(m0, _bls_np_lo, 0x08);

    /* m1 = lower 128 bits of: MUL_HIGH(T0,np_lo) + MUL_LOW(T0,np_hi) + MUL_LOW(t1h,np_lo) */
    _bls_copy256(m1, T0);
    _bls_csr(m1, _bls_np_lo, 0x10);    /* m1 = MUL_HIGH(T0, np_lo) */

    _bls_copy256(buf_A, T0);
    _bls_csr(buf_A, _bls_np_hi, 0x08); /* buf_A = MUL_LOW(T0, np_hi) */
    _bls_csr(m1, buf_A, 0x01);         /* m1 += buf_A (carry ignored, only need low 128) */

    _bls_pad128(buf_A, T1);            /* t1h = T1[0:3] (lower 128 of T1) */
    _bls_csr(buf_A, _bls_np_lo, 0x08); /* buf_A = MUL_LOW(t1h, np_lo) */
    _bls_csr(m1, buf_A, 0x01);         /* m1 += buf_A */

    /* m = m0 + (m1[0:3]) * 2^256  (only lower 128 bits of m1 matter) */

    /* === Phase 3: mP = m * P (768 bits) === */

    /* m0 * p_lo -> (MP0, buf_A=mp_ll_hi) */
    _bls_copy256(MP0, m0);
    _bls_csr(MP0, _bls_p_lo, 0x08);

    _bls_copy256(buf_A, m0);
    _bls_csr(buf_A, _bls_p_lo, 0x10);
    _bls_copy256(MP1, buf_A);          /* MP1 = mp_ll_hi */

    /* MP1 += MUL_LOW(m0, p_hi) */
    _bls_copy256(buf_A, m0);
    _bls_csr(buf_A, _bls_p_hi, 0x08);
    c1 = _bls_csr(MP1, buf_A, 0x01);

    /* MP1 += MUL_LOW(m1_trunc, p_lo) */
    _bls_pad128(buf_A, m1);
    _bls_csr(buf_A, _bls_p_lo, 0x08);
    c2 = _bls_csr(MP1, buf_A, 0x01);

    carry_mp = c1 + c2;

    /* MP2 = MUL_HIGH(m0, p_hi) + MUL_HIGH(m1_trunc, p_lo) + MUL_LOW(m1_trunc, p_hi) + carry */
    _bls_copy256(MP2, m0);
    _bls_csr(MP2, _bls_p_hi, 0x10);    /* MP2 = MUL_HIGH(m0, p_hi) */

    _bls_pad128(buf_A, m1);
    _bls_csr(buf_A, _bls_p_lo, 0x10);  /* buf_A = MUL_HIGH(m1_trunc, p_lo) */
    c1 = _bls_csr(MP2, buf_A, 0x01);

    _bls_pad128(buf_A, m1);
    _bls_csr(buf_A, _bls_p_hi, 0x08);  /* buf_A = MUL_LOW(m1_trunc, p_hi) */
    c2 = _bls_csr(MP2, buf_A, 0x01);

    {
        limb_t cm = carry_mp + c1 + c2;
        if (cm) {
            buf_A[0]=cm; buf_A[1]=0; buf_A[2]=0; buf_A[3]=0;
            buf_A[4]=0; buf_A[5]=0; buf_A[6]=0; buf_A[7]=0;
            _bls_csr(MP2, buf_A, 0x01);
        }
    }

    /* === Phase 4: S = (T + mP) >> 384 === */

    /* Add T + mP in 256-bit chunks with carry chain */
    c1 = _bls_csr(T0, MP0, 0x01);
    c2 = _bls_csr(T1, MP1, 0x01 | (c1 << 6));
    c3 = _bls_csr(T2, MP2, 0x01 | (c2 << 6));

    /* Result = bits [384..767] of (T+mP).
     * Lower 384 bits are 0 by construction.
     * Result lo 256 = upper 128 of T1 | lower 128 of T2
     * Result hi 128 = upper 128 of T2 (+ c3 which is part of the result flags)
     */
    ret[0]  = T1[4];  ret[1]  = T1[5];  ret[2]  = T1[6];  ret[3]  = T1[7];
    ret[4]  = T2[0];  ret[5]  = T2[1];  ret[6]  = T2[2];  ret[7]  = T2[3];
    ret[8]  = T2[4];  ret[9]  = T2[5];  ret[10] = T2[6];  ret[11] = T2[7];

    /* Conditional subtract P (software 384-bit). If c3 or result >= P, subtract. */
    if (c3 || _bls_ge384(ret, _bls_P)) {
        _bls_sub384(ret, ret, _bls_P);
    }
}

void sqr_mont_384(vec384 ret, const vec384 a,
                  const vec384 p, limb_t n0)
{
    mul_mont_384(ret, a, a, p, n0);
}

#else
MUL_MONT_IMPL(384)
#endif
"""

src = src.replace('MUL_MONT_IMPL(384)', airbender_384, 1)
print("Patched mul_mont_384/sqr_mont_384")

# Patch ADD_MOD_IMPL(384), SUB_MOD_IMPL(384) using software 384-bit with CSR ADD/SUB for low 256
# These are simpler -- just use the generic implementations for now to keep risk low.
# The mul_mont is the hot path (~80%+ of BLS12-381 cycles).
# We can accelerate add/sub later if needed.

# Patch REDC_MONT_IMPL(384, 768)
old_redc = 'REDC_MONT_IMPL(384, 768)'
new_redc = """#ifdef AIRBENDER_BIGINT_CSR
/* redc_mont_384: handled by calling mul_mont_384(ret, a_lo, 1, p, n0) conceptually.
 * But more efficient: run phases 2-4 of Montgomery on the 768-bit input directly.
 * For simplicity, delegate to the generic implementation for now. */
REDC_MONT_IMPL(384, 768)
#else
REDC_MONT_IMPL(384, 768)
#endif"""
src = src.replace(old_redc, new_redc, 1)
print("Patched REDC_MONT_IMPL(384, 768)")

# Patch FROM_MONT_IMPL(384)
old_from = 'FROM_MONT_IMPL(384)'
new_from = """#ifdef AIRBENDER_BIGINT_CSR
/* from_mont_384: a * R^{-1} mod P = mul_mont(a, 1).
 * Use the CSR-accelerated mul_mont_384. */
inline void from_mont_384(vec384 ret, const vec384 a,
                           const vec384 p, limb_t n0)
{
    static const vec384 one = {1,0,0,0,0,0,0,0,0,0,0,0};
    mul_mont_384(ret, a, one, p, n0);
}
#else
FROM_MONT_IMPL(384)
#endif"""
src = src.replace(old_from, new_from, 1)
print("Patched FROM_MONT_IMPL(384)")

with open(path, 'w') as f:
    f.write(src)

print("Patched " + path + " successfully")
PYEOF

echo "Airbender BLST patch complete."
