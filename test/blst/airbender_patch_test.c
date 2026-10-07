// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// SPDX-License-Identifier: Apache-2.0

// Host test of blst as cmake/patch_blst_airbender.sh patches it for the Airbender guest, which
// native builds otherwise never compile. The build runs the script on a copy of blst's vect.c,
// vect.h, bytes.h and no_asm.h; this file compiles them with a software model of the BigInt
// delegation (CSR 0x7CA) that enforces its operand rules, and checks the patched functions
// against a word-serial reference: the Fp2 multiply and square on coefficients up to 2p (edge,
// random, aliased and unaligned operands) and the 384-bit Montgomery multiply, addition and
// subtraction.
//
// Usage: blst_airbender_patch_test [SEED [ROUNDS]]

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __BLST_NO_ASM__
#define AIRBENDER_BIGINT_CSR
#define AIRBENDER_BIGINT_CSR_MODEL

// The patched code is written for the guest's 32-bit limbs. blst's vect.h picks 64-bit limbs on
// the 64-bit architectures it knows and 32-bit ones in a no-asm build for any other, so hide the
// host architecture from blst's sources (the system headers they include are already included).
#pragma push_macro("__x86_64__")
#pragma push_macro("__aarch64__")
#undef __x86_64__
#undef __aarch64__
#include "vect.h"
#include "bytes.h"
#include "vect.c"
#pragma pop_macro("__aarch64__")
#pragma pop_macro("__x86_64__")

_Static_assert(LIMB_T_BITS == 32 && sizeof(limb_t) == 4, "the patched code needs 32-bit limbs");

#define FP_WORDS 12

// The delegation as riscv_transpiler (vm/delegations/bigint.rs) implements it, for the operations
// the patch uses. The transpiler asserts 32-byte aligned and distinct operands, and the patch
// never relies on a SUB borrow-in. Operands in ROM cannot be told apart on the host.
enum
{
    OP_ADD = 0x01,
    OP_SUB = 0x02,
    OP_MUL_LOW = 0x08,
    OP_MUL_HIGH = 0x10,
    OP_CARRY_IN = 0x40,
    OP_MEMCOPY = 0x80
};

static void model_misuse(const char* what, const limb_t* mut, const limb_t* immut, limb_t mask)
{
    fprintf(stderr, "BigInt delegation misuse: %s (x10=%p x11=%p mask=0x%02x)\n", what,
        (const void*)mut, (const void*)immut, (unsigned)mask);
    abort();
}

limb_t _bls_csr(limb_t* mut, const limb_t* immut, limb_t mask)
{
    uint32_t a[8], b[8], r[16];
    uint64_t acc;
    int i, j;

    if ((uintptr_t)mut % 32 != 0 || (uintptr_t)immut % 32 != 0)
        model_misuse("operand not 32-byte aligned", mut, immut, mask);
    if (mut == immut)
        model_misuse("x10 == x11", mut, immut, mask);
    memcpy(a, mut, sizeof(a));
    memcpy(b, immut, sizeof(b));

    switch (mask)
    {
    case OP_ADD:
    case OP_ADD | OP_CARRY_IN:
        acc = mask >> 6;
        for (i = 0; i < 8; ++i)
        {
            acc += (uint64_t)a[i] + b[i];
            mut[i] = (uint32_t)acc;
            acc >>= 32;
        }
        return (limb_t)acc;
    case OP_SUB:
        acc = 0;
        for (i = 0; i < 8; ++i)
        {
            acc = (uint64_t)a[i] - b[i] - acc;
            mut[i] = (uint32_t)acc;
            acc >>= 63;
        }
        return (limb_t)acc;
    case OP_MUL_LOW:
    case OP_MUL_HIGH:
        memset(r, 0, sizeof(r));
        for (i = 0; i < 8; ++i)
        {
            acc = 0;
            for (j = 0; j < 8; ++j)
            {
                acc += (uint64_t)a[i] * b[j] + r[i + j];
                r[i + j] = (uint32_t)acc;
                acc >>= 32;
            }
            r[i + 8] = (uint32_t)acc;
        }
        memcpy(mut, mask == OP_MUL_LOW ? r : r + 8, 32);
        if (mask == OP_MUL_HIGH)
            return 0;
        for (i = 8; i < 16; ++i)  // MUL_LOW reports whether the product exceeds 256 bits
        {
            if (r[i] != 0)
                return 1;
        }
        return 0;
    case OP_MEMCOPY:
        memcpy(mut, b, sizeof(b));
        return 0;
    default:
        model_misuse(mask == (OP_SUB | OP_CARRY_IN) ? "SUB with borrow-in" : "unexpected mask", mut,
            immut, mask);
        return 0;
    }
}

// Reference arithmetic, independent of blst and of the patch: p comes from its hex digits, and
// the Montgomery product is a word-serial REDC followed by subtractions of p until it is below p.
static limb_t P[FP_WORDS], P2[FP_WORDS], P_INV;  // p, 2p, -1/p mod 2^32

static void from_hex(limb_t r[FP_WORDS], const char* hex)
{
    size_t n = strlen(hex), i;
    memset(r, 0, FP_WORDS * sizeof(limb_t));
    for (i = 0; i < n; ++i)
    {
        const char c = hex[n - 1 - i];
        const limb_t d = (limb_t)(c <= '9' ? c - '0' : c - 'a' + 10);
        r[i / 8] |= d << (4 * (i % 8));
    }
}

static int ref_cmp(const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    int i;
    for (i = FP_WORDS - 1; i >= 0; --i)
    {
        if (a[i] != b[i])
            return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

static limb_t ref_add(limb_t r[FP_WORDS], const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    uint64_t acc = 0;
    int i;
    for (i = 0; i < FP_WORDS; ++i)
    {
        acc += (uint64_t)a[i] + b[i];
        r[i] = (limb_t)acc;
        acc >>= 32;
    }
    return (limb_t)acc;
}

static limb_t ref_sub(limb_t r[FP_WORDS], const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    uint64_t acc = 0;
    int i;
    for (i = 0; i < FP_WORDS; ++i)
    {
        acc = (uint64_t)a[i] - b[i] - acc;
        r[i] = (limb_t)acc;
        acc >>= 63;
    }
    return (limb_t)acc;
}

static void ref_add_mod(limb_t r[FP_WORDS], const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    if (ref_add(r, a, b) || ref_cmp(r, P) >= 0)
        ref_sub(r, r, P);
}

static void ref_sub_mod(limb_t r[FP_WORDS], const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    if (ref_sub(r, a, b))
        ref_add(r, r, P);
}

// r = a*b/2^384 mod p, fully reduced, for any a and b below 2^384.
static void ref_mont_mul(limb_t r[FP_WORDS], const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    limb_t t[2 * FP_WORDS + 1] = {0};
    uint64_t acc;
    int i, j;

    for (i = 0; i < FP_WORDS; ++i)
    {
        acc = 0;
        for (j = 0; j < FP_WORDS; ++j)
        {
            acc += (uint64_t)a[i] * b[j] + t[i + j];
            t[i + j] = (limb_t)acc;
            acc >>= 32;
        }
        t[i + FP_WORDS] = (limb_t)acc;
    }
    for (i = 0; i < FP_WORDS; ++i)
    {
        const limb_t m = t[i] * P_INV;
        acc = 0;
        for (j = 0; j < FP_WORDS; ++j)
        {
            acc += (uint64_t)m * P[j] + t[i + j];
            t[i + j] = (limb_t)acc;
            acc >>= 32;
        }
        for (j = i + FP_WORDS; acc != 0; ++j)
        {
            acc += t[j];
            t[j] = (limb_t)acc;
            acc >>= 32;
        }
    }
    // The quotient is below 2^384 + p.
    while (t[2 * FP_WORDS] != 0 || ref_cmp(t + FP_WORDS, P) >= 0)
        t[2 * FP_WORDS] -= ref_sub(t + FP_WORDS, t + FP_WORDS, P);
    memcpy(r, t + FP_WORDS, FP_WORDS * sizeof(limb_t));
}

// Schoolbook Fp2 product (four products, unlike the Karatsuba and lazy forms).
static void ref_fp2_mul(vec384x r, const vec384x a, const vec384x b)
{
    limb_t t0[FP_WORDS], t1[FP_WORDS], t2[FP_WORDS], t3[FP_WORDS];
    ref_mont_mul(t0, a[0], b[0]);
    ref_mont_mul(t1, a[1], b[1]);
    ref_mont_mul(t2, a[0], b[1]);
    ref_mont_mul(t3, a[1], b[0]);
    ref_sub_mod(r[0], t0, t1);
    ref_add_mod(r[1], t2, t3);
}

static uint64_t rng_state;

static limb_t rnd32(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (limb_t)((rng_state * 0x2545f4914f6cdd1dULL) >> 32);
}

// Random value below bound, drawn from the bits of bound's length.
static void rnd_below(limb_t r[FP_WORDS], const limb_t bound[FP_WORDS])
{
    int top = FP_WORDS - 1, i;
    limb_t mask;
    while (bound[top] == 0)
        --top;
    for (mask = bound[top]; mask & (mask + 1); mask |= mask >> 1)
        ;
    do
    {
        memset(r, 0, FP_WORDS * sizeof(limb_t));
        for (i = 0; i < top; ++i)
            r[i] = rnd32();
        r[top] = rnd32() & mask;
    } while (ref_cmp(r, bound) >= 0);
}

#define MAX_EDGES 32
static limb_t edges[MAX_EDGES][FP_WORDS];
static int n_edges;

static limb_t* new_edge(void)
{
    if (n_edges == MAX_EDGES)
        abort();
    return memset(edges[n_edges++], 0, FP_WORDS * sizeof(limb_t));
}

// Values at the carry, borrow and range boundaries, all below 2p.
static void make_edges(void)
{
    limb_t one[FP_WORDS] = {1}, two[FP_WORDS] = {2}, three[FP_WORDS] = {3};
    limb_t low_ones[FP_WORDS] = {0}, word8[FP_WORDS] = {0}, *e;
    int i;

    for (i = 0; i < 8; ++i)
        low_ones[i] = 0xffffffff;  // 2^256 - 1
    word8[8] = 1;                  // 2^256

    new_edge();
    memcpy(new_edge(), one, sizeof(one));
    memcpy(new_edge(), two, sizeof(two));
    ref_sub(new_edge(), P, one);
    ref_sub(new_edge(), P, two);
    memcpy(new_edge(), P, sizeof(P));
    ref_add(new_edge(), P, one);
    ref_sub(new_edge(), P2, one);
    ref_sub(new_edge(), P2, two);
    ref_sub(new_edge(), P2, three);
    e = new_edge();  // (p - 1)/2
    for (i = 0; i < FP_WORDS; ++i)
        e[i] = P[i] >> 1 | (i + 1 < FP_WORDS ? P[i + 1] << 31 : 0);
    ref_add(new_edge(), e, one);  // (p + 1)/2
    ref_add(new_edge(), e, P);    // (3p - 1)/2
    memcpy(new_edge(), low_ones, sizeof(low_ones));
    memcpy(new_edge(), word8, sizeof(word8));
    ref_add(new_edge(), word8, one);
    ref_sub(new_edge(), P, word8);
    ref_add(new_edge(), P, low_ones);
    e = new_edge();  // p with its low 256 bits cleared, then set
    memcpy(e + 8, P + 8, 4 * sizeof(limb_t));
    ref_add(new_edge(), e, low_ones);
    e = new_edge();  // 2^128 - 1
    memset(e, 0xff, 4 * sizeof(limb_t));
    new_edge()[11] = 0x10000000;  // 2^380
    e = new_edge();               // 2^381 - 1
    memset(e, 0xff, 11 * sizeof(limb_t));
    e[11] = 0x1fffffff;
    new_edge()[11] = 0x20000000;  // 2^381
    e = new_edge();               // 2^384 mod p, the Montgomery one
    ref_sub(e, e, P);
    while (ref_cmp(e, P) >= 0)
        ref_sub(e, e, P);

    for (i = 0; i < n_edges; ++i)
    {
        if (ref_cmp(edges[i], P2) >= 0)
            abort();
    }
}

enum
{
    CANONICAL,  // [0, p)
    ABOVE_P,    // [p, 2p)
    BELOW_2P,   // [0, 2p)
    EDGE,
    SPARSE,  // one to three random or all-ones words, below 2p
    N_KINDS
};

static void rnd_coeff(limb_t r[FP_WORDS], int kind)
{
    int i, n, w;
    switch (kind)
    {
    case CANONICAL:
        rnd_below(r, P);
        break;
    case ABOVE_P:
        rnd_below(r, P);
        ref_add(r, r, P);
        break;
    case BELOW_2P:
        rnd_below(r, P2);
        break;
    case EDGE:
        memcpy(r, edges[rnd32() % (limb_t)n_edges], FP_WORDS * sizeof(limb_t));
        break;
    default:
        do
        {
            memset(r, 0, FP_WORDS * sizeof(limb_t));
            for (i = 0, n = 1 + (int)(rnd32() % 3); i < n; ++i)
            {
                w = (int)(rnd32() % FP_WORDS);
                r[w] = rnd32() & 1 ? 0xffffffff : rnd32();
            }
        } while (ref_cmp(r, P2) >= 0);
        break;
    }
}

// Operands live at every word offset from a 32-byte boundary, like blst's limb-aligned values.
static limb_t pool[3][2 * FP_WORDS + 8] __attribute__((aligned(32)));
static unsigned placement;
static long cases, failures;

static limb_t (*place(int slot, const vec384x v))[FP_WORDS]
{
    limb_t(*p)[FP_WORDS] = (limb_t(*)[FP_WORDS])(pool[slot] + (placement + 3 * (unsigned)slot) % 8);
    memcpy(p, v, sizeof(vec384x));
    return p;
}

static void print_fp(const char* name, const limb_t a[FP_WORDS])
{
    int i;
    fprintf(stderr, "  %s = 0x", name);
    for (i = FP_WORDS - 1; i >= 0; --i)
        fprintf(stderr, "%08x", (unsigned)a[i]);
    fprintf(stderr, "\n");
}

static void expect_fp(const char* what, const vec384 got, const vec384 want, const vec384 a,
    const vec384 b)
{
    ++cases;
    if (memcmp(got, want, sizeof(vec384)) == 0 || ++failures > 10)
        return;
    fprintf(stderr, "%s: wrong result\n", what);
    print_fp("a", a);
    print_fp("b", b);
    print_fp("got", got);
    print_fp("want", want);
}

static void expect_fp2(const char* what, const vec384x got, const vec384x want, const vec384x a,
    const vec384x b)
{
    ++cases;
    if (memcmp(got, want, sizeof(vec384x)) == 0 || ++failures > 10)
        return;
    fprintf(stderr, "%s: wrong result\n", what);
    print_fp("a0", a[0]);
    print_fp("a1", a[1]);
    print_fp("b0", b[0]);
    print_fp("b1", b[1]);
    print_fp("got0", got[0]);
    print_fp("got1", got[1]);
    print_fp("want0", want[0]);
    print_fp("want1", want[1]);
}

static void check_fp2_mul(const vec384x a, const vec384x b)
{
    vec384x want, r;
    limb_t(*pa)[FP_WORDS], (*pb)[FP_WORDS], (*pr)[FP_WORDS];

    ++placement;
    ref_fp2_mul(want, a, b);
    pa = place(0, a);
    pb = place(1, b);
    pr = place(2, a);
    mul_mont_384x(pr, pa, pb, P, P_INV);
    memcpy(r, pr, sizeof(r));
    expect_fp2("mul_mont_384x", r, want, a, b);
    mul_mont_384x(pa, pa, pb, P, P_INV);
    memcpy(r, pa, sizeof(r));
    expect_fp2("mul_mont_384x, ret == a", r, want, a, b);
    pa = place(0, a);
    mul_mont_384x(pb, pa, pb, P, P_INV);
    memcpy(r, pb, sizeof(r));
    expect_fp2("mul_mont_384x, ret == b", r, want, a, b);
}

static void check_fp2_sqr(const vec384x a)
{
    vec384x want, r;
    limb_t(*pa)[FP_WORDS], (*pr)[FP_WORDS];

    ++placement;
    ref_fp2_mul(want, a, a);
    pa = place(0, a);
    pr = place(2, a);
    sqr_mont_384x(pr, pa, P, P_INV);
    memcpy(r, pr, sizeof(r));
    expect_fp2("sqr_mont_384x", r, want, a, a);
    sqr_mont_384x(pa, pa, P, P_INV);
    memcpy(r, pa, sizeof(r));
    expect_fp2("sqr_mont_384x, ret == a", r, want, a, a);
    pa = place(0, a);
    mul_mont_384x(pr, pa, pa, P, P_INV);
    memcpy(r, pr, sizeof(r));
    expect_fp2("mul_mont_384x, a == b", r, want, a, a);
    mul_mont_384x(pa, pa, pa, P, P_INV);
    memcpy(r, pa, sizeof(r));
    expect_fp2("mul_mont_384x, ret == a == b", r, want, a, a);
}

// The base field operations, on coefficients below 2p for the multiply and below p otherwise.
static void check_fp(const limb_t a[FP_WORDS], const limb_t b[FP_WORDS])
{
    vec384 want, r, ra, rb;

    ref_mont_mul(want, a, b);
    mul_mont_384(r, a, b, P, P_INV);
    expect_fp("mul_mont_384", r, want, a, b);
    if (ref_cmp(a, P) >= 0 || ref_cmp(b, P) >= 0)
        return;
    memcpy(ra, a, sizeof(ra));
    memcpy(rb, b, sizeof(rb));
    ref_add_mod(want, a, b);
    add_mod_384(ra, ra, b, P);
    expect_fp("add_mod_384", ra, want, a, b);
    ref_sub_mod(want, a, b);
    sub_mod_384(rb, a, rb, P);
    expect_fp("sub_mod_384", rb, want, a, b);
}

// Known answers from an arbitrary-precision implementation, with coefficients in [p, 2p).
static void check_known_answers(void)
{
    vec384x a, b, want, r;
    from_hex(a[0], "340223d472ffcd3496374f6c869759aec8ee9709e70a257e"
                   "ce61a541ed61ec483d57fffd62a7ffff73fdffffffff5555");
    from_hex(a[1], "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf"
                   "6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaae");
    from_hex(b[0], "1a0111ea397fe69a4b1ba7b6434bacd864774b84f38512bf"
                   "6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
    from_hex(b[1], "340223d472ffcd3496374f6c869759aec8ee9709e70a247e"
                   "ce61a541ed61ec483d57fffd62a7ffff73fdffffffff5556");

    from_hex(want[0], "10c623c0995beb3b48dfb5202999562d4e99aa8c1cd62ad9"
                      "34d2888801d0d9d2928f2ed2f0ec9fd2dac01ba68319b7ac");
    from_hex(want[1], "0a9bc3199e0d1859e44d9195ec6484f624e5b8051ca2a806"
                      "a98c516476f838cf5b371519c5595b7195dcd0b67578db5b");
    ref_fp2_mul(r, a, b);
    expect_fp2("reference Fp2 multiply", r, want, a, b);
    check_fp2_mul(a, b);

    from_hex(want[0], "0e1145584aa6e6e8a2ce74c1f4971d8b0285b7ffe35f13f2"
                      "3060769e1a2b4cb8d7ab6c501484281e6f5ced363fa369ad");
    from_hex(want[1], "040caf87a99d3387e753eda3e69e6af268c6771eada60a45"
                      "ca7c244e55f43c019a15913c630e1e16e505f1e8afbaa497");
    ref_fp2_mul(r, a, a);
    expect_fp2("reference Fp2 square", r, want, a, a);
    check_fp2_sqr(a);
}

int main(int argc, char* argv[])
{
    const uint64_t seed = argc > 1 ? strtoull(argv[1], NULL, 0) : 0x41b5a7e5;
    const long rounds = argc > 2 ? strtol(argv[2], NULL, 0) : 20000;
    vec384x a, b;
    long n;
    int i, j, k, l;

    from_hex(P, "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf"
                "6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
    ref_add(P2, P, P);
    for (P_INV = 1, i = 0; i < 5; ++i)  // Newton: each step doubles the correct low bits
        P_INV *= 2 - P[0] * P_INV;
    P_INV = 0 - P_INV;
    make_edges();
    rng_state = seed | 1;

    check_known_answers();

    // The extremes of the bounds in patch_blst_airbender.sh: the most negative a0*b0 - a1*b1,
    // the largest products, the most negative a0 - a1 + p and the largest (a0 + a1)*(a0 - a1 + p).
    for (i = 0; i < n_edges; ++i)
    {
        for (j = 0; j < n_edges; ++j)
        {
            memcpy(a[0], edges[i], sizeof(a[0]));
            memcpy(a[1], edges[j], sizeof(a[1]));
            check_fp2_sqr(a);
            for (k = 0; k < 4; ++k)
            {
                for (l = 0; l < 2; ++l)
                    memcpy(b[l], edges[rnd32() % (limb_t)n_edges], sizeof(b[l]));
                check_fp2_mul(a, b);
            }
            check_fp(edges[i], edges[j]);
        }
    }

    for (n = 0; n < rounds; ++n)
    {
        for (l = 0; l < 2; ++l)
        {
            rnd_coeff(a[l], (int)(rnd32() % N_KINDS));
            rnd_coeff(b[l], (int)(rnd32() % N_KINDS));
        }
        check_fp2_mul(a, b);
        check_fp2_sqr(a);
        check_fp(a[0], b[0]);

        // a1 above a0 + p, where the square adds p twice
        rnd_below(a[0], P);
        do
            rnd_coeff(a[1], ABOVE_P);
        while (ref_sub(b[0], a[1], a[0]) || ref_cmp(b[0], P) <= 0);
        check_fp2_sqr(a);
    }

    printf("blst Airbender patch: %ld checks, %ld failures (seed 0x%" PRIx64 ")\n", cases,
        failures, seed);
    return failures != 0;
}
