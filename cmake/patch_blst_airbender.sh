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

# Patch ADD_MOD_IMPL(384) to use CSR ADD/SUB for the low 256-bit chunk.
old_add_384 = 'ADD_MOD_IMPL(384)'
new_add_384 = r"""#ifdef AIRBENDER_BIGINT_CSR
/* add_mod_384: ret = (a + b) mod p, using CSR ADD for low 256 bits.
 * Split 384-bit values into lo(256) + hi(128). CSR handles lo, manual carry for hi.
 * 1. tmp_lo = CSR ADD(a_lo, b_lo), carry_lo
 * 2. tmp_hi = a_hi + b_hi + carry_lo (manual 4-word chain), carry384
 * 3. sub_lo = CSR SUB(tmp_lo, p_lo), borrow_lo
 * 4. sub_hi = tmp_hi - p_hi - borrow_lo (manual), borrow384
 * 5. If carry384 - borrow384 == 0 (subtraction succeeded): ret = sub, else ret = tmp
 */
inline void add_mod_384(vec384 ret, const vec384 a,
                        const vec384 b, const vec384 p)
{
    (void)p;
    limb_t tmp_lo[8] _BLS_ALIGN32;
    limb_t sub_lo[8] _BLS_ALIGN32;
    limb_t tmp_hi[4], sub_hi[4];
    limb_t carry_lo, borrow_lo, carry384, borrow384;
    llimb_t limbx;

    /* Step 1: tmp_lo = a_lo + b_lo via CSR ADD */
    tmp_lo[0]=a[0]; tmp_lo[1]=a[1]; tmp_lo[2]=a[2]; tmp_lo[3]=a[3];
    tmp_lo[4]=a[4]; tmp_lo[5]=a[5]; tmp_lo[6]=a[6]; tmp_lo[7]=a[7];
    sub_lo[0]=b[0]; sub_lo[1]=b[1]; sub_lo[2]=b[2]; sub_lo[3]=b[3];
    sub_lo[4]=b[4]; sub_lo[5]=b[5]; sub_lo[6]=b[6]; sub_lo[7]=b[7];
    carry_lo = _bls_csr(tmp_lo, sub_lo, 0x01);  /* tmp_lo = a_lo + b_lo */

    /* Step 2: tmp_hi = a_hi + b_hi + carry_lo */
    limbx = (llimb_t)a[8]  + b[8]  + carry_lo;
    tmp_hi[0] = (limb_t)limbx; carry384 = (limb_t)(limbx >> LIMB_T_BITS);
    limbx = (llimb_t)a[9]  + b[9]  + carry384;
    tmp_hi[1] = (limb_t)limbx; carry384 = (limb_t)(limbx >> LIMB_T_BITS);
    limbx = (llimb_t)a[10] + b[10] + carry384;
    tmp_hi[2] = (limb_t)limbx; carry384 = (limb_t)(limbx >> LIMB_T_BITS);
    limbx = (llimb_t)a[11] + b[11] + carry384;
    tmp_hi[3] = (limb_t)limbx; carry384 = (limb_t)(limbx >> LIMB_T_BITS);

    /* Step 3: sub_lo = tmp_lo - p_lo via CSR SUB */
    _bls_copy256(sub_lo, tmp_lo);
    borrow_lo = _bls_csr(sub_lo, _bls_p_lo, 0x02);  /* sub_lo = tmp_lo - p_lo */

    /* Step 4: sub_hi = tmp_hi - p_hi - borrow_lo */
    limbx = (llimb_t)tmp_hi[0] - _bls_P[8]  - borrow_lo;
    sub_hi[0] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)tmp_hi[1] - _bls_P[9]  - borrow384;
    sub_hi[1] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)tmp_hi[2] - _bls_P[10] - borrow384;
    sub_hi[2] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)tmp_hi[3] - _bls_P[11] - borrow384;
    sub_hi[3] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;

    /* Step 5: Select. If carry384 >= borrow384, subtraction succeeded: use sub.
     * Otherwise (carry384==0, borrow384==1): use tmp (a+b < p). */
    if (carry384 - borrow384) {
        /* Subtraction underflowed: use tmp (a+b) */
        ret[0]=tmp_lo[0]; ret[1]=tmp_lo[1]; ret[2]=tmp_lo[2]; ret[3]=tmp_lo[3];
        ret[4]=tmp_lo[4]; ret[5]=tmp_lo[5]; ret[6]=tmp_lo[6]; ret[7]=tmp_lo[7];
        ret[8]=tmp_hi[0]; ret[9]=tmp_hi[1]; ret[10]=tmp_hi[2]; ret[11]=tmp_hi[3];
    } else {
        /* Subtraction succeeded: use sub (a+b-p) */
        ret[0]=sub_lo[0]; ret[1]=sub_lo[1]; ret[2]=sub_lo[2]; ret[3]=sub_lo[3];
        ret[4]=sub_lo[4]; ret[5]=sub_lo[5]; ret[6]=sub_lo[6]; ret[7]=sub_lo[7];
        ret[8]=sub_hi[0]; ret[9]=sub_hi[1]; ret[10]=sub_hi[2]; ret[11]=sub_hi[3];
    }
}
#else
ADD_MOD_IMPL(384)
#endif"""
src = src.replace(old_add_384, new_add_384, 1)
print("Patched add_mod_384 with CSR ADD/SUB")

# Patch SUB_MOD_IMPL(384) to use CSR SUB/ADD for the low 256-bit chunk.
old_sub_384 = 'SUB_MOD_IMPL(384)'
new_sub_384 = r"""#ifdef AIRBENDER_BIGINT_CSR
/* sub_mod_384: ret = (a - b) mod p, using CSR SUB for low 256 bits.
 * 1. ret_lo = CSR SUB(a_lo, b_lo), borrow_lo
 * 2. ret_hi = a_hi - b_hi - borrow_lo (manual), borrow384
 * 3. If borrow384: ret_lo = CSR ADD(ret_lo, p_lo), carry_lo
 *                  ret_hi += p_hi + carry_lo (manual)
 */
inline void sub_mod_384(vec384 ret, const vec384 a,
                        const vec384 b, const vec384 p)
{
    (void)p;
    limb_t ret_lo[8] _BLS_ALIGN32;
    limb_t b_lo[8] _BLS_ALIGN32;
    limb_t ret_hi[4];
    limb_t borrow_lo, borrow384, carry_lo, carry;
    llimb_t limbx;

    /* Step 1: ret_lo = a_lo - b_lo via CSR SUB */
    ret_lo[0]=a[0]; ret_lo[1]=a[1]; ret_lo[2]=a[2]; ret_lo[3]=a[3];
    ret_lo[4]=a[4]; ret_lo[5]=a[5]; ret_lo[6]=a[6]; ret_lo[7]=a[7];
    b_lo[0]=b[0]; b_lo[1]=b[1]; b_lo[2]=b[2]; b_lo[3]=b[3];
    b_lo[4]=b[4]; b_lo[5]=b[5]; b_lo[6]=b[6]; b_lo[7]=b[7];
    borrow_lo = _bls_csr(ret_lo, b_lo, 0x02);  /* ret_lo = a_lo - b_lo */

    /* Step 2: ret_hi = a_hi - b_hi - borrow_lo */
    limbx = (llimb_t)a[8]  - b[8]  - borrow_lo;
    ret_hi[0] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)a[9]  - b[9]  - borrow384;
    ret_hi[1] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)a[10] - b[10] - borrow384;
    ret_hi[2] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;
    limbx = (llimb_t)a[11] - b[11] - borrow384;
    ret_hi[3] = (limb_t)limbx; borrow384 = (limb_t)(limbx >> LIMB_T_BITS) & 1;

    /* Step 3: If borrow, add p back */
    if (borrow384) {
        carry_lo = _bls_csr(ret_lo, _bls_p_lo, 0x01);  /* ret_lo += p_lo */

        limbx = (llimb_t)ret_hi[0] + _bls_P[8]  + carry_lo;
        ret_hi[0] = (limb_t)limbx; carry = (limb_t)(limbx >> LIMB_T_BITS);
        limbx = (llimb_t)ret_hi[1] + _bls_P[9]  + carry;
        ret_hi[1] = (limb_t)limbx; carry = (limb_t)(limbx >> LIMB_T_BITS);
        limbx = (llimb_t)ret_hi[2] + _bls_P[10] + carry;
        ret_hi[2] = (limb_t)limbx; carry = (limb_t)(limbx >> LIMB_T_BITS);
        limbx = (llimb_t)ret_hi[3] + _bls_P[11] + carry;
        ret_hi[3] = (limb_t)limbx;
    }

    ret[0]=ret_lo[0]; ret[1]=ret_lo[1]; ret[2]=ret_lo[2]; ret[3]=ret_lo[3];
    ret[4]=ret_lo[4]; ret[5]=ret_lo[5]; ret[6]=ret_lo[6]; ret[7]=ret_lo[7];
    ret[8]=ret_hi[0]; ret[9]=ret_hi[1]; ret[10]=ret_hi[2]; ret[11]=ret_hi[3];
}
#else
SUB_MOD_IMPL(384)
#endif"""
src = src.replace(old_sub_384, new_sub_384, 1)
print("Patched sub_mod_384 with CSR SUB/ADD")

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

# Patch mul_mont_384x: redirect mul_mont_n(... NLIMBS(384)) -> mul_mont_384
# The generic mul_mont_n is O(n^2) scalar CIOS; mul_mont_384 uses BigInt CSR.
old_mul384x = """void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b,
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
new_mul384x = """void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b,
                          const vec384 p, limb_t n0)
{
    vec384 aa, bb, cc;

#ifdef AIRBENDER_BIGINT_CSR
    add_mod_384(aa, a[0], a[1], p);
    add_mod_384(bb, b[0], b[1], p);
    mul_mont_384(bb, bb, aa, p, n0);
    mul_mont_384(aa, a[0], b[0], p, n0);
    mul_mont_384(cc, a[1], b[1], p, n0);
    sub_mod_384(ret[0], aa, cc, p);
    sub_mod_384(ret[1], bb, aa, p);
    sub_mod_384(ret[1], ret[1], cc, p);
#else
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
src = src.replace(old_mul384x, new_mul384x, 1)
print("Patched mul_mont_384x -> mul_mont_384 + add/sub_mod_384")

# Patch sgn0_pty_mont_384: redirect from_mont_n -> from_mont_384
old_sgn0_384 = """inline limb_t sgn0_pty_mont_384(const vec384 a, const vec384 p, limb_t n0)
{
    vec384 tmp;

    from_mont_n(tmp, a, p, n0, NLIMBS(384));

    return sgn0_pty_mod_n(tmp, p, NLIMBS(384));
}"""
new_sgn0_384 = """inline limb_t sgn0_pty_mont_384(const vec384 a, const vec384 p, limb_t n0)
{
    vec384 tmp;

#ifdef AIRBENDER_BIGINT_CSR
    from_mont_384(tmp, a, p, n0);
#else
    from_mont_n(tmp, a, p, n0, NLIMBS(384));
#endif

    return sgn0_pty_mod_n(tmp, p, NLIMBS(384));
}"""
src = src.replace(old_sgn0_384, new_sgn0_384, 1)
print("Patched sgn0_pty_mont_384 -> from_mont_384")

# Patch sgn0_pty_mont_384x: redirect from_mont_n -> from_mont_384
old_sgn0_384x = """inline limb_t sgn0_pty_mont_384x(const vec384x a, const vec384 p, limb_t n0)
{
    vec384x tmp;

    from_mont_n(tmp[0], a[0], p, n0, NLIMBS(384));
    from_mont_n(tmp[1], a[1], p, n0, NLIMBS(384));

    return sgn0_pty_mod_384x(tmp, p);
}"""
new_sgn0_384x = """inline limb_t sgn0_pty_mont_384x(const vec384x a, const vec384 p, limb_t n0)
{
    vec384x tmp;

#ifdef AIRBENDER_BIGINT_CSR
    from_mont_384(tmp[0], a[0], p, n0);
    from_mont_384(tmp[1], a[1], p, n0);
#else
    from_mont_n(tmp[0], a[0], p, n0, NLIMBS(384));
    from_mont_n(tmp[1], a[1], p, n0, NLIMBS(384));
#endif

    return sgn0_pty_mod_384x(tmp, p);
}"""
src = src.replace(old_sgn0_384x, new_sgn0_384x, 1)
print("Patched sgn0_pty_mont_384x -> from_mont_384")

# Patch sqr_n_mul_mont_383: redirect mul_mont_nonred_n/mul_mont_n -> mul_mont_384
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
#ifdef AIRBENDER_BIGINT_CSR
    __builtin_assume(count != 0);
    while(count--) {
        mul_mont_384(ret, a, a, p, n0);
        a = ret;
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
src = src.replace(old_sqr_n_mul, new_sqr_n_mul, 1)
print("Patched sqr_n_mul_mont_383 -> mul_mont_384")

# Patch sqr_mont_382x: redirect mul_mont_nonred_n -> mul_mont_384
old_sqr_382x = """    /* "mul_mont_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));

    /* "add_mod_n(ret[1], ret[1], ret[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limb_t a_i = ret[1][i];
        ret[1][i] = a_i<<1 | carry;
        carry = a_i>>(LIMB_T_BITS-1);
    }

    /* "mul_mont_n(ret[0], t0, t1, p, n0, NLIMBS(384));" */
    mul_mont_nonred_n(ret[0], t0, t1, p, n0, NLIMBS(384));"""
new_sqr_382x = """    /* "mul_mont_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));" */
#ifdef AIRBENDER_BIGINT_CSR
    mul_mont_384(ret[1], a[0], a[1], p, n0);
#else
    mul_mont_nonred_n(ret[1], a[0], a[1], p, n0, NLIMBS(384));
#endif

    /* "add_mod_n(ret[1], ret[1], ret[1], p, NLIMBS(384));" */
    for (carry=0, i=0; i<NLIMBS(384); i++) {
        limb_t a_i = ret[1][i];
        ret[1][i] = a_i<<1 | carry;
        carry = a_i>>(LIMB_T_BITS-1);
    }

    /* "mul_mont_n(ret[0], t0, t1, p, n0, NLIMBS(384));" */
#ifdef AIRBENDER_BIGINT_CSR
    mul_mont_384(ret[0], t0, t1, p, n0);
#else
    mul_mont_nonred_n(ret[0], t0, t1, p, n0, NLIMBS(384));
#endif"""
src = src.replace(old_sqr_382x, new_sqr_382x, 1)
print("Patched sqr_mont_382x -> mul_mont_384")

with open(path, 'w') as f:
    f.write(src)

print("Patched " + path + " successfully")
PYEOF

echo "Airbender BLST patch complete."
