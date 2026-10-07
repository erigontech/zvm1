// ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2018 Pawel Bylica (original)
// SPDX-License-Identifier: Apache-2.0

#include "keccak.h"

#include <stdint.h>

#ifdef SP1TURBO
void syscall_keccak_permute(uint64_t (*state)[25]);
#elif defined(SP1)
static inline __attribute__((always_inline)) void syscall_keccak_permute(uint64_t state[25])
{
    register uint64_t t0 asm("t0") = 0x00010109;
    register uint64_t* a0 asm("a0") = state;
    register uint64_t a1 asm("a1") = 0;
    asm volatile("ecall" : "+r"(t0) : "r"(a0), "r"(a1) : "memory");
}
#elif defined(AIRBENDER)
// File-level static buffer for keccak CSR delegation (256-byte aligned).
// Shared between syscall_keccak_permute and direct-access optimized paths.
static uint64_t __attribute__((aligned(256))) buf[32];

// 32-byte-aligned zero source for CSR 0x7CA MEMCOPY-based bulk zeroing.
// Each MEMCOPY(dst, zeros, 0x80) clears 32 bytes (= 4 uint64_t) in 4 insns,
// replacing 8 sw-zero stores that the scalar loop emits.
// Not const: must be in RAM (.bss), not .rodata (ROM), because CSR requires x11 in RAM.
static uint64_t __attribute__((aligned(32))) keccak_zeros[4] = {0, 0, 0, 0};

/// Zero buf[0..31] (256 bytes) via 8 CSR MEMCOPY calls from keccak_zeros.
/// Replaces scalar loop (~82 insns: 62 sw + 20 loop overhead) with ~40 insns.
/// Single asm block keeps x11 (source) pinned across all 8 calls; x10 (dest) advances.
static inline __attribute__((always_inline))
void buf_zero_all(void)
{
    const unsigned long src = (unsigned long)keccak_zeros;
    const unsigned long dst = (unsigned long)buf;
    __asm__ __volatile__(
        "mv x11, %[src]\n\t"
        // chunk 0: buf[0..3]
        "mv x10, %[dst]\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 1: buf[4..7]
        "addi x10, %[dst], 32\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 2: buf[8..11]
        "addi x10, %[dst], 64\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 3: buf[12..15]
        "addi x10, %[dst], 96\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 4: buf[16..19]
        "addi x10, %[dst], 128\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 5: buf[20..23]
        "addi x10, %[dst], 160\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 6: buf[24..27]
        "addi x10, %[dst], 192\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        // chunk 7: buf[28..31]
        "addi x10, %[dst], 224\n\t"
        "li x12, 0x80\n\t"
        "csrrw x0, 0x7CA, x0\n\t"
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "x10", "x11", "x12", "memory"
    );
}

/// Zeroes the state bytes of buf[] from byte offset @p off (a multiple of 4, at most 200) by
/// jumping into a run of word stores.
///
/// The delegation reads and writes buf[0..30], but only the 25 lanes of the state need a clean
/// start: every round writes the six scratch lanes buf[25..30] before it reads them (the column
/// XORs, then the column mix), and buf[31] is never touched. A short input leaves 16 to 43 words
/// to clear; 1 store each beats the CSR MEMCOPY zeroing (4 instructions and a delegation per
/// 32-byte chunk, plus a per-chunk loop and word stores up to the next 32-byte boundary).
static inline __attribute__((always_inline)) void buf_zero_state_from(size_t off)
{
    uintptr_t t;
    __asm__(
        "lla %[t], 1f\n\t"
        "add %[t], %[t], %[off]\n\t"
        "jr %[t]\n"
        "1:\n\t"
        ".set .Lkz_off, 0\n\t"
        ".rept 50\n\t"
        "sw zero, .Lkz_off(%[b])\n\t"
        ".set .Lkz_off, .Lkz_off + 4\n\t"
        ".endr"
        // The lanes' own type, so GCC keeps the later padding-bit OR into a lane after this.
        : [t] "=&r"(t), "+m"(*(uint64_t(*)[25])buf)
        : [off] "r"(off), [b] "r"(buf));
}

/// Keccak-f[1600] via airbender CSR 0x7CB delegation, permuting the 256-byte-aligned state s in place.
/// 649 consecutive CSR writes — the transpiler's preprocess_bytecode
/// scans for exactly 649 contiguous csrrw instructions.
static inline __attribute__((always_inline)) void keccak_permute_at(uint64_t* s)
{
    register uint32_t ctrl __asm__("x10") = 0;
    register void*    sptr __asm__("x11") = (void*)s;
    __asm__ __volatile__(
        ".rept 649\n"
        "  csrrw x0, 0x7CB, x0\n"
        ".endr\n"
        : "+r"(ctrl)
        : "r"(sptr)
        : "memory"
    );
}

/// The same on buf[]. The delegation takes any 256-byte-aligned state in RAM, so the snapshot pool
/// of ethash_keccak256_resume() is permuted in place by keccak_permute_at() too.
static inline __attribute__((always_inline)) void keccak_permute_buf(void)
{
    keccak_permute_at(buf);
}

/// Keccak-f[1600] on any state. keccak() keeps its state in buf[] itself, so once inlined there
/// the copies fold away; any other state goes through buf[].
static inline __attribute__((always_inline)) void syscall_keccak_permute(uint64_t state[25])
{
    int i;
    if (state != buf)
    {
        // Zero buf[25..30] before copying state in.
        for (i = 25; i < 31; i++)
            buf[i] = 0;
        for (i = 0; i < 25; i++)
            buf[i] = state[i];
    }
    keccak_permute_buf();
    if (state != buf)
    {
        for (i = 0; i < 25; i++)
            state[i] = buf[i];
    }
}
#endif

// Provide __has_attribute macro if not defined.
#ifndef __has_attribute
#define __has_attribute(name) 0
#endif

// Provide __has_builtin macro if not defined.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// [[always_inline]]
#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

/**
 * Whether to clear the state with explicit stores rather than leave the spelling to the compiler.
 *
 * `uint64_t state[25] = {0}` is a 200-byte clear. On RISC-V the compiler has no cheap inline
 * expansion for it and emits a call to memset -- one call per hash. That holds for both rv64, where
 * 25 stores replace it, and rv32, where 50 do and the call is the more expensive of the two.
 * Where the initializer is expanded inline already, as on x86-64, there is nothing to gain and the
 * plain form is kept; RISC-V is what this was measured on, so RISC-V is what is named here.
 */
#if defined(__riscv)
#define KECCAK_INLINE_STATE_CLEAR 1
#else
#define KECCAK_INLINE_STATE_CLEAR 0
#endif

#if !__has_builtin(__builtin_memcpy) && !defined(__GNUC__)
#include <string.h>
#define __builtin_memcpy memcpy
#endif

// [[noinline]]
#if defined(_MSC_VER)
#define NO_INLINE __declspec(noinline)
#elif __has_attribute(noinline)
#define NO_INLINE __attribute__((noinline))
#else
#define NO_INLINE
#endif

// Excludes a function from AddressSanitizer instrumentation.
#if defined(_MSC_VER)
#define NO_SANITIZE_ADDRESS __declspec(no_sanitize_address)
#elif __has_attribute(no_sanitize_address)
#define NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define NO_SANITIZE_ADDRESS
#endif

#define WORD_SIZE sizeof(uint64_t)
#define WORD_BITS 64

/**
 * Whether the target needs aligned loads, so that reading the input as aligned words pays.
 *
 * Where an unaligned load is a single instruction, load_le_word() already compiles to it and
 * assembling words out of aligned ones only adds work: measured up to +3.6% per hash on x86-64.
 */
#ifndef KECCAK_STRICT_ALIGNMENT  // Overridable so the tests can exercise both paths anywhere.
#if defined(__riscv) || defined(__mips__) || defined(__sparc__) || defined(__hppa__)
#define KECCAK_STRICT_ALIGNMENT 1
#else
#define KECCAK_STRICT_ALIGNMENT 0
#endif
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define to_le64(X) __builtin_bswap64(X)
#else
#define to_le64(X) X
#endif

/// Loads a 64-bit little-endian integer, a single load instruction on any target. Under
/// KECCAK_STRICT_ALIGNMENT the caller must guarantee the alignment: declaring it is what keeps
/// this one instruction there, since the copy would otherwise be lowered byte by byte.
static inline ALWAYS_INLINE uint64_t load_le_word(const uint8_t* data)
{
#if KECCAK_STRICT_ALIGNMENT && __has_builtin(__builtin_assume_aligned)
    data = (const uint8_t*)__builtin_assume_aligned(data, WORD_SIZE);
#endif
    uint64_t word;
    __builtin_memcpy(&word, data, sizeof(word));
    return to_le64(word);
}

#if defined(AIRBENDER)
/// Loads a 64-bit little-endian integer from any address. load_le_word() may assume 8-byte
/// alignment on RISC-V; here 4-byte aligned inputs take two word loads (~4 insns), anything
/// else falls back to the copy, which -mstrict-align lowers byte by byte.
static inline ALWAYS_INLINE uint64_t load_le_any(const uint8_t* data)
{
    if (__builtin_expect(((uintptr_t)data & 3) == 0, 1))
    {
        const uint32_t* w = (const uint32_t*)data;
        return to_le64((uint64_t)w[0] | ((uint64_t)w[1] << 32));
    }
    uint64_t word;
    __builtin_memcpy(&word, data, sizeof(word));
    return to_le64(word);
}
#endif

#if defined(AIRBENDER)
typedef uint32_t __attribute__((may_alias)) keccak_word32;

/// The word holding the last 0-3 input bytes and the padding byte after them.
static inline ALWAYS_INLINE uint32_t tail_word(const uint8_t* t, size_t r)
{
    if (r == 0)
        return 1;
    uint32_t w = (uint32_t)1 << (8 * r);
    w |= t[0];
    if (r >= 2)
        w |= (uint32_t)t[1] << 8;
    if (r == 3)
        w |= (uint32_t)t[2] << 16;
    return w;
}

/// XORs the last incomplete block (rem < 136 bytes, 4-aligned data) and the padding byte into s.
static inline ALWAYS_INLINE void absorb_last_aligned(keccak_word32* s, const uint8_t* data, size_t rem)
{
    const keccak_word32* const d = (const keccak_word32*)data;
    const size_t m = rem / 4;
    switch (m)
    {
    case 33: s[32] ^= d[32]; /* fallthrough */
    case 32: s[31] ^= d[31]; /* fallthrough */
    case 31: s[30] ^= d[30]; /* fallthrough */
    case 30: s[29] ^= d[29]; /* fallthrough */
    case 29: s[28] ^= d[28]; /* fallthrough */
    case 28: s[27] ^= d[27]; /* fallthrough */
    case 27: s[26] ^= d[26]; /* fallthrough */
    case 26: s[25] ^= d[25]; /* fallthrough */
    case 25: s[24] ^= d[24]; /* fallthrough */
    case 24: s[23] ^= d[23]; /* fallthrough */
    case 23: s[22] ^= d[22]; /* fallthrough */
    case 22: s[21] ^= d[21]; /* fallthrough */
    case 21: s[20] ^= d[20]; /* fallthrough */
    case 20: s[19] ^= d[19]; /* fallthrough */
    case 19: s[18] ^= d[18]; /* fallthrough */
    case 18: s[17] ^= d[17]; /* fallthrough */
    case 17: s[16] ^= d[16]; /* fallthrough */
    case 16: s[15] ^= d[15]; /* fallthrough */
    case 15: s[14] ^= d[14]; /* fallthrough */
    case 14: s[13] ^= d[13]; /* fallthrough */
    case 13: s[12] ^= d[12]; /* fallthrough */
    case 12: s[11] ^= d[11]; /* fallthrough */
    case 11: s[10] ^= d[10]; /* fallthrough */
    case 10: s[9] ^= d[9]; /* fallthrough */
    case 9: s[8] ^= d[8]; /* fallthrough */
    case 8: s[7] ^= d[7]; /* fallthrough */
    case 7: s[6] ^= d[6]; /* fallthrough */
    case 6: s[5] ^= d[5]; /* fallthrough */
    case 5: s[4] ^= d[4]; /* fallthrough */
    case 4: s[3] ^= d[3]; /* fallthrough */
    case 3: s[2] ^= d[2]; /* fallthrough */
    case 2: s[1] ^= d[1]; /* fallthrough */
    case 1: s[0] ^= d[0]; /* fallthrough */
    case 0:
        break;
    default:
        __builtin_unreachable();
    }
    s[m] ^= tail_word(data + 4 * m, rem % 4);
}

/// Copies a short input (size < 136, 4-aligned data) with its padding byte into the zero state s.
/// Returns the number of words written.
static inline ALWAYS_INLINE size_t copy_short_aligned(keccak_word32* s, const uint8_t* data, size_t size)
{
    const keccak_word32* const d = (const keccak_word32*)data;
    const size_t m = size / 4;
    switch (m)
    {
    case 33: s[32] = d[32]; /* fallthrough */
    case 32: s[31] = d[31]; /* fallthrough */
    case 31: s[30] = d[30]; /* fallthrough */
    case 30: s[29] = d[29]; /* fallthrough */
    case 29: s[28] = d[28]; /* fallthrough */
    case 28: s[27] = d[27]; /* fallthrough */
    case 27: s[26] = d[26]; /* fallthrough */
    case 26: s[25] = d[25]; /* fallthrough */
    case 25: s[24] = d[24]; /* fallthrough */
    case 24: s[23] = d[23]; /* fallthrough */
    case 23: s[22] = d[22]; /* fallthrough */
    case 22: s[21] = d[21]; /* fallthrough */
    case 21: s[20] = d[20]; /* fallthrough */
    case 20: s[19] = d[19]; /* fallthrough */
    case 19: s[18] = d[18]; /* fallthrough */
    case 18: s[17] = d[17]; /* fallthrough */
    case 17: s[16] = d[16]; /* fallthrough */
    case 16: s[15] = d[15]; /* fallthrough */
    case 15: s[14] = d[14]; /* fallthrough */
    case 14: s[13] = d[13]; /* fallthrough */
    case 13: s[12] = d[12]; /* fallthrough */
    case 12: s[11] = d[11]; /* fallthrough */
    case 11: s[10] = d[10]; /* fallthrough */
    case 10: s[9] = d[9]; /* fallthrough */
    case 9: s[8] = d[8]; /* fallthrough */
    case 8: s[7] = d[7]; /* fallthrough */
    case 7: s[6] = d[6]; /* fallthrough */
    case 6: s[5] = d[5]; /* fallthrough */
    case 5: s[4] = d[4]; /* fallthrough */
    case 4: s[3] = d[3]; /* fallthrough */
    case 3: s[2] = d[2]; /* fallthrough */
    case 2: s[1] = d[1]; /* fallthrough */
    case 1: s[0] = d[0]; /* fallthrough */
    case 0:
        break;
    default:
        __builtin_unreachable();
    }
    s[m] = tail_word(data + 4 * m, size % 4);
    return m + 1;
}
#endif

#if KECCAK_INLINE_STATE_CLEAR
/// Clears the state with 25 stores.
///
/// The unrolling is load bearing, not a micro-optimization of the loop itself: left as a loop this
/// is a 200-byte clear again, and the call comes back. Unrolled there is no loop left to recognize,
/// with GCC and clang alike and at every optimization level. The call it replaces is newlib's
/// memset doing about 80 instructions for this clear, on every hash.
static inline ALWAYS_INLINE void clear_state(uint64_t state[25])
{
    size_t i;
#pragma GCC unroll 25
    for (i = 0; i < 25; ++i)
        state[i] = 0;
}
#endif

/// Reads the absorbed input as 64-bit little-endian words assembled from aligned loads, one load
/// per word: a word of an unaligned input spans two, but one is carried over from the previous.
/// The two ends of an unaligned input are read past, by under a word -- see absorb_input_unaligned.
struct word_reader
{
    const uint8_t* data;  ///< The position of the next word of the input.
    uint64_t carry;       ///< The already loaded leading bytes of the next word.
    unsigned misalign;    ///< The misalignment of the input in bytes, 0 if it is aligned.
};

/// Starts reading the input of @p size bytes at @p data. @p misalign must be its misalignment,
/// (uintptr_t)data % WORD_SIZE: every load in the reader is aligned only because of that.
static inline ALWAYS_INLINE struct word_reader init_word_reader(
    const uint8_t* data, size_t size, unsigned misalign)
{
    struct word_reader r = {data, 0, misalign};
    if (misalign != 0 && size >= WORD_SIZE)
    {
        // The leading bytes of the first word are the trailing bytes of the aligned word before it.
        // A shorter input has no whole word to read, so the carry would go unused -- and for an
        // empty one this would be the single load not sharing its word with any input byte.
        r.carry = load_le_word(data - misalign) >> (misalign * 8);
    }
    return r;
}

/// Returns the next 64-bit little-endian word of an unaligned input and advances the reader. A
/// whole word must be left to read, and the misalignment must not be zero: the shift would be 64.
static inline ALWAYS_INLINE uint64_t read_unaligned_word(struct word_reader* r)
{
    const unsigned shift = r->misalign * 8;
    const uint64_t loaded = load_le_word(r->data + WORD_SIZE - r->misalign);
    r->data += WORD_SIZE;

    // The loaded word's leading bytes complete this word, its trailing ones the next.
    const uint64_t word = r->carry | (loaded << (WORD_BITS - shift));
    r->carry = loaded >> shift;
    return word;
}

/// Absorbs the next @p words words of the input into the state's leading lanes. The alignment is
/// tested here rather than per word to keep the body small enough for a whole block's absorb to
/// stay fully unrolled; an earlier, larger one lost the unrolling and cost 3.1% of total cycles.
static inline ALWAYS_INLINE void absorb_words(uint64_t* state, size_t words, struct word_reader* r)
{
    if (r->misalign == 0)
    {
        const uint8_t* const data = r->data;
        for (size_t i = 0; i < words; ++i)
            state[i] ^= load_le_word(data + i * WORD_SIZE);
        r->data = data + words * WORD_SIZE;
    }
    else
    {
        for (size_t i = 0; i < words; ++i)
            state[i] ^= read_unaligned_word(r);
    }
}

/// Rotates the bits of x left by the count value specified by s.
/// The s must be in range <0, 64> exclusively, otherwise the result is undefined.
static inline uint64_t rol(uint64_t x, unsigned s)
{
    return (x << s) | (x >> (64 - s));
}

static const uint64_t round_constants[24] = {  //
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};


/// The Keccak-f[1600] function.
///
/// The implementation of the Keccak-f function with 1600-bit width of the permutation (b).
/// The size of the state is also 1600 bit what gives 25 64-bit words.
///
/// @param state  The state of 25 64-bit words on which the permutation is to be performed.
///
/// The implementation based on:
/// - "simple" implementation by Ronny Van Keer, included in "Reference and optimized code in C",
///   https://keccak.team/archives.html, CC0-1.0 / Public Domain.
static inline ALWAYS_INLINE void keccakf1600_implementation(uint64_t state[25])
{
    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;

    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    uint64_t Ba, Be, Bi, Bo, Bu;

    uint64_t Da, De, Di, Do, Du;

    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (size_t n = 0; n < 24; n += 2)
    {
        // Round (n + 0): Axx -> Exx

        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        Da = Bu ^ rol(Be, 1);
        De = Ba ^ rol(Bi, 1);
        Di = Be ^ rol(Bo, 1);
        Do = Bi ^ rol(Bu, 1);
        Du = Bo ^ rol(Ba, 1);

        Ba = Aba ^ Da;
        Be = rol(Age ^ De, 44);
        Bi = rol(Aki ^ Di, 43);
        Bo = rol(Amo ^ Do, 21);
        Bu = rol(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ round_constants[n];
        Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu);
        Ebo = Bo ^ (~Bu & Ba);
        Ebu = Bu ^ (~Ba & Be);

        Ba = rol(Abo ^ Do, 28);
        Be = rol(Agu ^ Du, 20);
        Bi = rol(Aka ^ Da, 3);
        Bo = rol(Ame ^ De, 45);
        Bu = rol(Asi ^ Di, 61);
        Ega = Ba ^ (~Be & Bi);
        Ege = Be ^ (~Bi & Bo);
        Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba);
        Egu = Bu ^ (~Ba & Be);

        Ba = rol(Abe ^ De, 1);
        Be = rol(Agi ^ Di, 6);
        Bi = rol(Ako ^ Do, 25);
        Bo = rol(Amu ^ Du, 8);
        Bu = rol(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi);
        Eke = Be ^ (~Bi & Bo);
        Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba);
        Eku = Bu ^ (~Ba & Be);

        Ba = rol(Abu ^ Du, 27);
        Be = rol(Aga ^ Da, 36);
        Bi = rol(Ake ^ De, 10);
        Bo = rol(Ami ^ Di, 15);
        Bu = rol(Aso ^ Do, 56);
        Ema = Ba ^ (~Be & Bi);
        Eme = Be ^ (~Bi & Bo);
        Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba);
        Emu = Bu ^ (~Ba & Be);

        Ba = rol(Abi ^ Di, 62);
        Be = rol(Ago ^ Do, 55);
        Bi = rol(Aku ^ Du, 39);
        Bo = rol(Ama ^ Da, 41);
        Bu = rol(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi);
        Ese = Be ^ (~Bi & Bo);
        Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba);
        Esu = Bu ^ (~Ba & Be);


        // Round (n + 1): Exx -> Axx

        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        Da = Bu ^ rol(Be, 1);
        De = Ba ^ rol(Bi, 1);
        Di = Be ^ rol(Bo, 1);
        Do = Bi ^ rol(Bu, 1);
        Du = Bo ^ rol(Ba, 1);

        Ba = Eba ^ Da;
        Be = rol(Ege ^ De, 44);
        Bi = rol(Eki ^ Di, 43);
        Bo = rol(Emo ^ Do, 21);
        Bu = rol(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ round_constants[n + 1];
        Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu);
        Abo = Bo ^ (~Bu & Ba);
        Abu = Bu ^ (~Ba & Be);

        Ba = rol(Ebo ^ Do, 28);
        Be = rol(Egu ^ Du, 20);
        Bi = rol(Eka ^ Da, 3);
        Bo = rol(Eme ^ De, 45);
        Bu = rol(Esi ^ Di, 61);
        Aga = Ba ^ (~Be & Bi);
        Age = Be ^ (~Bi & Bo);
        Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba);
        Agu = Bu ^ (~Ba & Be);

        Ba = rol(Ebe ^ De, 1);
        Be = rol(Egi ^ Di, 6);
        Bi = rol(Eko ^ Do, 25);
        Bo = rol(Emu ^ Du, 8);
        Bu = rol(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi);
        Ake = Be ^ (~Bi & Bo);
        Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba);
        Aku = Bu ^ (~Ba & Be);

        Ba = rol(Ebu ^ Du, 27);
        Be = rol(Ega ^ Da, 36);
        Bi = rol(Eke ^ De, 10);
        Bo = rol(Emi ^ Di, 15);
        Bu = rol(Eso ^ Do, 56);
        Ama = Ba ^ (~Be & Bi);
        Ame = Be ^ (~Bi & Bo);
        Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba);
        Amu = Bu ^ (~Ba & Be);

        Ba = rol(Ebi ^ Di, 62);
        Be = rol(Ego ^ Do, 55);
        Bi = rol(Eku ^ Du, 39);
        Bo = rol(Ema ^ Da, 41);
        Bu = rol(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi);
        Ase = Be ^ (~Bi & Bo);
        Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba);
        Asu = Bu ^ (~Ba & Be);
    }

    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}

static void keccakf1600_generic(uint64_t state[25])
{
    keccakf1600_implementation(state);
}

/// The pointer to the best Keccak-f[1600] function implementation,
/// selected during runtime initialization.
#if defined(SP1TURBO) || defined(SP1) || defined(AIRBENDER)
/// Not a pointer: GCC will not inline the syscall through one, leaving a call and a return
/// around the four instructions of the ecall. Nothing selects another implementation here.
#define keccakf1600_best syscall_keccak_permute
#else
#define DEFAULT_keccakf1600 keccakf1600_generic

static void (*keccakf1600_best)(uint64_t[25]) = DEFAULT_keccakf1600;
#endif


#if !defined(_MSC_VER) && defined(__x86_64__) && __has_attribute(target)
__attribute__((target("bmi,bmi2"))) static void keccakf1600_bmi(uint64_t state[25])
{
    keccakf1600_implementation(state);
}

__attribute__((constructor)) static void select_keccakf1600_implementation(void)
{
    // Init CPU information.
    // This is needed on macOS because of the bug: https://bugs.llvm.org/show_bug.cgi?id=48459.
    __builtin_cpu_init();

    // Check if both BMI and BMI2 are supported. Some CPUs like Intel E5-2697 v2 incorrectly
    // report BMI2 but not BMI being available.
    if (__builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2"))
        keccakf1600_best = keccakf1600_bmi;
}
#endif


/// Absorbs the whole input, permuting after every complete block, and XORs in the padding byte.
/// The last block's bit flip and the final permutation are left to the caller. Pass a nonzero
/// @p misalign only through absorb_input_unaligned(), whose ASan exclusion covers its reads.
static inline ALWAYS_INLINE void absorb_input(
    uint64_t* state, size_t block_words, const uint8_t* data, size_t size, unsigned misalign)
{
    struct word_reader reader = init_word_reader(data, size, misalign);

    while (size >= block_words * WORD_SIZE)
    {
        absorb_words(state, block_words, &reader);

        keccakf1600_best(state);

        size -= block_words * WORD_SIZE;
    }

#if defined(AIRBENDER)
    if (misalign == 0)
    {
        absorb_last_aligned((keccak_word32*)state, reader.data, size);
        return;
    }
#endif

    const size_t last_words = size / WORD_SIZE;  // Whole words of the last, incomplete block.
    absorb_words(state, last_words, &reader);
    size %= WORD_SIZE;

    // Absorb last 0–7 bytes of input + the padding byte.
    const uint8_t* const tail = reader.data;
    uint64_t last_word = (uint64_t)0x01 << (size * 8);
    for (size_t i = 0; i < size; ++i)
        last_word |= (uint64_t)tail[i] << (i * 8);
    state[last_words] ^= last_word;
}

/// Absorbs an unaligned input, out of line so the aligned case need not save the registers this
/// wants, and so the reads past the input land here: they cannot fault (an aligned load stays in
/// the page of the input byte it shares a word with) and are shifted out, but ASan objects.
static NO_INLINE NO_SANITIZE_ADDRESS void absorb_input_unaligned(
    uint64_t* state, size_t block_words, const uint8_t* data, size_t size, unsigned misalign)
{
    absorb_input(state, block_words, data, size, misalign);
}

static inline ALWAYS_INLINE void keccak(
    uint64_t* out, size_t bits, const uint8_t* data, size_t size)
{
    const size_t hash_size = bits / 8;
    const size_t block_words = (1600 - bits * 2) / 8 / WORD_SIZE;

#if defined(AIRBENDER)
    // The state is the CSR-aligned static buf[], so every permutation runs in place, without the
    // state→buf→state copies around the delegation.
    uint64_t* const state = buf;
    if (size >= block_words * WORD_SIZE && ((uintptr_t)data & 3) == 0)
    {
        // The state starts at zero, so absorbing the first block is a copy: store the block and
        // zero the rest of the state, rather than zeroing everything (8 CSR MEMCOPY delegations)
        // and then XOR-ing the block in, which reloads the zero state word by word. The scratch
        // lanes past the state need no clearing (see buf_zero_state_from()).
        const uint32_t* const s = (const uint32_t*)data;
        uint32_t* const d = (uint32_t*)buf;
        size_t i;
#pragma GCC unroll 34
        for (i = 0; i < 2 * block_words; ++i)
            d[i] = s[i];
#pragma GCC unroll 32
        for (; i < 2 * 25; ++i)
            d[i] = 0;
        keccak_permute_buf();
        // A block is a multiple of 8 bytes, so the rest of the input keeps its alignment.
        data += block_words * WORD_SIZE;
        size -= block_words * WORD_SIZE;
    }
    else
        buf_zero_all();
#elif KECCAK_INLINE_STATE_CLEAR
    uint64_t state[25];
    clear_state(state);
#else
    uint64_t state[25] = {0};
#endif

    // Off the strict targets this is a constant 0, folding the word reader down to a plain load.
    const unsigned misalign =
        KECCAK_STRICT_ALIGNMENT ? (unsigned)((uintptr_t)data % WORD_SIZE) : 0;
    if (misalign == 0)
        absorb_input(state, block_words, data, size, 0);
    else
        absorb_input_unaligned(state, block_words, data, size, misalign);

    state[block_words - 1] ^= 0x8000000000000000;  // Last block bit flip.

    keccakf1600_best(state);

    for (size_t i = 0; i < (hash_size / WORD_SIZE); ++i)
        out[i] = to_le64(state[i]);
}

union ethash_hash256 ethash_keccak256(const uint8_t* data, size_t size)
{
    union ethash_hash256 hash;
#if defined(AIRBENDER)
    // For keccak-256: block_size = (1600 - 256*2) / 8 = 136 bytes.
    // Most EVM inputs are < 136 bytes (single block). Specialize.
    if (size < 136)
    {
        // Direct copy to CSR-aligned buf: skip buf_zero_all() + XOR loop since
        // XOR-with-zero is identity. Copy data via uint32_t when aligned (~2x faster
        // than load_le + XOR), then zero only the remaining buf words.
        {
            int i;
            uint32_t* bufW = (uint32_t*)buf;
            const uint8_t* d = data;
            size_t remaining = size;

            // Fast path: copy full 4-byte words when data is 4-byte aligned.
            if (__builtin_expect(((uintptr_t)d & 3) == 0, 1))
            {
                const size_t words = copy_short_aligned((keccak_word32*)buf, data, size);
                buf_zero_state_from(4 * words);
                buf[16] |= 0x8000000000000000ULL;
                keccak_permute_buf();
                hash.word64s[0] = to_le64(buf[0]);
                hash.word64s[1] = to_le64(buf[1]);
                hash.word64s[2] = to_le64(buf[2]);
                hash.word64s[3] = to_le64(buf[3]);
                return hash;
            }
            if (0)
            {
                const uint32_t* dW = (const uint32_t*)d;
                size_t full_words = remaining / 4;
                for (size_t j = 0; j < full_words; ++j)
                    bufW[j] = dW[j];
                d += full_words * 4;
                bufW += full_words;
                remaining -= full_words * 4;
            }
            else
            {
                // Unaligned: use load_le for full uint64_t chunks.
                uint64_t* buf_iter = buf;
                while (remaining >= 8)
                {
                    *buf_iter++ = load_le_any(d);
                    d += 8;
                    remaining -= 8;
                }
                bufW = (uint32_t*)buf_iter;
            }

            // Handle remaining bytes + padding byte 0x01.
            uint64_t last_word = 0;
            uint8_t* lw = (uint8_t*)&last_word;
            for (i = 0; i < (int)remaining; ++i)
                lw[i] = d[i];
            lw[remaining] = 0x01;
            {
                // Write last_word at current position (may be uint32_t-misaligned).
                uint32_t* lwd = (uint32_t*)&last_word;
                bufW[0] = lwd[0];
                bufW[1] = lwd[1];
                bufW += 2;
            }

            // Zero the rest of the state (the scratch lanes need no clearing).
            buf_zero_state_from((size_t)((uintptr_t)bufW - (uintptr_t)buf));

            buf[16] |= 0x8000000000000000ULL;

            keccak_permute_buf();
            hash.word64s[0] = to_le64(buf[0]);
            hash.word64s[1] = to_le64(buf[1]);
            hash.word64s[2] = to_le64(buf[2]);
            hash.word64s[3] = to_le64(buf[3]);
            return hash;
        }
    }
#endif
    keccak(hash.word64s, 256, data, size);
    return hash;
}

/// Copies the 25 lanes of the state in buf[] to the pool slot. Word by word and unrolled: a loop
/// GCC recognizes as a copy becomes a memcpy call, and the guest's memcpy is the BigInt delegation.
#if defined(AIRBENDER)
static inline ALWAYS_INLINE void save_buf_lanes(uint64_t* slot)
{
    const keccak_word32* const s = (const keccak_word32*)buf;
    keccak_word32* const d = (keccak_word32*)slot;
    size_t i;
#pragma GCC unroll 50
    for (i = 0; i < 2 * 25; ++i)
        d[i] = s[i];
}
#endif

union ethash_hash256 ethash_keccak256_snap(
    const uint8_t* data, size_t size, size_t blocks, uint64_t* slot)
{
    union ethash_hash256 hash;
    const size_t block_words = (1600 - 256 * 2) / 8 / WORD_SIZE;
#if defined(AIRBENDER)
    // keccak() with a copy of the state after every block, kept when it is the blocks-th. The state
    // lives in buf[] as there: the snapshot is the only extra work.
    {
        const keccak_word32* const s = (const keccak_word32*)data;
        keccak_word32* const d = (keccak_word32*)buf;
        size_t i;
#pragma GCC unroll 34
        for (i = 0; i < 2 * block_words; ++i)
            d[i] = s[i];
#pragma GCC unroll 32
        for (; i < 2 * 25; ++i)
            d[i] = 0;
    }
    size_t done = 1;
    for (;;)
    {
        keccak_permute_buf();
        data += block_words * WORD_SIZE;
        size -= block_words * WORD_SIZE;
        if (done == blocks)
            save_buf_lanes(slot);
        if (size < block_words * WORD_SIZE)
            break;
        struct word_reader reader = {data, 0, 0};
        absorb_words(buf, block_words, &reader);
        ++done;
    }
    absorb_last_aligned((keccak_word32*)buf, data, size);
    buf[block_words - 1] ^= 0x8000000000000000;
    keccak_permute_buf();
    for (size_t i = 0; i < 4; ++i)
        hash.word64s[i] = to_le64(buf[i]);
#else
    uint64_t state[25] = {0};
    struct word_reader reader = {data, 0, 0};
    for (size_t b = 0; b < blocks; ++b)
    {
        absorb_words(state, block_words, &reader);
        keccakf1600_best(state);
    }
    for (size_t i = 0; i < 25; ++i)
        slot[i] = state[i];
    absorb_input(state, block_words, reader.data, size - blocks * block_words * WORD_SIZE, 0);
    state[block_words - 1] ^= 0x8000000000000000;
    keccakf1600_best(state);
    for (size_t i = 0; i < 4; ++i)
        hash.word64s[i] = to_le64(state[i]);
#endif
    return hash;
}

union ethash_hash256 ethash_keccak256_resume(
    uint64_t* slot, size_t blocks, const uint8_t* data, size_t size)
{
    union ethash_hash256 hash;
    const size_t block_words = (1600 - 256 * 2) / 8 / WORD_SIZE;
    data += blocks * block_words * WORD_SIZE;
    size -= blocks * block_words * WORD_SIZE;
#if defined(AIRBENDER)
    // The slot is permuted where it is, and consumed. Lanes 25..30 of it are the delegation's
    // scratch and need no initialization (see buf_zero_state_from()).
    while (size >= block_words * WORD_SIZE)
    {
        struct word_reader reader = {data, 0, 0};
        absorb_words(slot, block_words, &reader);
        keccak_permute_at(slot);
        data += block_words * WORD_SIZE;
        size -= block_words * WORD_SIZE;
    }
    absorb_last_aligned((keccak_word32*)slot, data, size);
    slot[block_words - 1] ^= 0x8000000000000000;
    keccak_permute_at(slot);
#else
    absorb_input(slot, block_words, data, size, 0);
    slot[block_words - 1] ^= 0x8000000000000000;
    keccakf1600_best(slot);
#endif
    for (size_t i = 0; i < 4; ++i)
        hash.word64s[i] = to_le64(slot[i]);
    return hash;
}

union ethash_hash256 ethash_keccak256_32(const uint8_t data[32])
{
    union ethash_hash256 hash;
#if defined(AIRBENDER)
    // Write directly to the CSR-aligned static buffer — skip the state→buf→state copies.
    {
        // Bulk-zero buf via CSR MEMCOPY (32 insns) then write data + padding.
        // Replaces scalar loops (~60 sw + overhead) with 8 CSR calls.
        buf_zero_all();
        buf[0] = load_le_any(data);
        buf[1] = load_le_any(data + 8);
        buf[2] = load_le_any(data + 16);
        buf[3] = load_le_any(data + 24);
        buf[4] = 0x0000000000000001ULL;
        buf[16] = 0x8000000000000000ULL;

        keccak_permute_buf();
        hash.word64s[0] = to_le64(buf[0]);
        hash.word64s[1] = to_le64(buf[1]);
        hash.word64s[2] = to_le64(buf[2]);
        hash.word64s[3] = to_le64(buf[3]);
        return hash;
    }
#else
    keccak(hash.word64s, 256, data, 32);
    return hash;
#endif
}

#if defined(AIRBENDER)
/// The number of index bits of keccak64_memo[]: 16384 slots, 2 MB. A block of the 200-block corpus
/// hashes 1280 distinct 64-byte inputs on average, and 11010 at most.
#define KECCAK64_MEMO_BITS 14

/// A 64-byte input of ethash_keccak256_64_be() and its result, padded to 128 bytes: a power of two
/// takes the index to the slot with one shift instead of three instructions for 96 bytes.
struct keccak64_memo_slot
{
    uint32_t key[16];   ///< The input words.
    uint32_t value[8];  ///< The result words. The slot is empty while value[0] is 0.
    uint32_t unused[8];
};

/// Direct-mapped memo of the 64-byte KECCAK256 inputs: the mapping slots keccak(key . slot), which
/// a contract hashes again on every access to the same entry. 65% of the 64-byte hashes in the
/// 200-block corpus repeat one done before in the block.
///
/// Zero-initialized, so it is .bss, which costs nothing at startup: Airbender RAM starts zeroed.
/// A slot answers only if its first result word is nonzero, so the zeroed table never answers for
/// the all-zero input before that has been hashed, and a result whose first word is 0 (one in
/// 2^32) is just recomputed every time. An answer compares all 64 bytes; collisions only evict.
static struct keccak64_memo_slot keccak64_memo[1u << KECCAK64_MEMO_BITS];

/// Hashes the 64 bytes at @p data into @p slot and copies the result to @p out: the memo's miss.
/// Out of line so that the permutation's fixed registers (x10 and x11, where the arguments arrive)
/// leave the hit path's register allocation alone.
static NO_INLINE void keccak64_memo_fill(
    ethash_w32 out[8], const ethash_w32* data, struct keccak64_memo_slot* slot)
{
    // The state of ethash_keccak256()'s short path for a 64-byte input: the input, the padding
    // byte, zeros, and the last block bit, which lane 16 holds alone.
    ethash_w32* const state = (ethash_w32*)buf;
    size_t i;
    // The input is memory in the word layout: each word holds the big-endian number of its 4
    // bytes, where the state takes them as the little-endian number. The key stays as it is, the
    // hit path compares and indexes the words as they come. Byte copies take two instructions a
    // byte; the barrier keeps GCC from merging them into shifts to swap, which take more.
    for (i = 0; i < 16; ++i)
    {
        const uint32_t w = data[i];
        slot->key[i] = w;
        const uint8_t* b = (const uint8_t*)&data[i];
        __asm__("" : "+r"(b));
        uint8_t* const sb = (uint8_t*)&state[i];
        sb[0] = b[3];
        sb[1] = b[2];
        sb[2] = b[1];
        sb[3] = b[0];
    }
    state[16] = 0x01;
    // The offset is fixed here, so plain stores do without buf_zero_state_from()'s computed jump.
#pragma GCC unroll 33
    for (i = 17; i < 2 * 25; ++i)
        state[i] = 0;
    buf[16] = 0x8000000000000000ULL;
    keccak_permute_buf();

    // The hash bytes reversed are the big-endian number's little-endian bytes. Byte copies take two
    // instructions a byte; the barriers keep GCC from merging them into halfword loads and stores
    // with shifts to swap, which take more.
    const uint8_t* h = (const uint8_t*)buf;
    uint8_t* const v = (uint8_t*)slot->value;
    __asm__("" : "+r"(h));
#pragma GCC unroll 32
    for (i = 0; i < 32; ++i)
    {
        uint8_t c = h[31 - i];
        __asm__("" : "+r"(c));
        v[i] = c;
    }
    for (i = 0; i < 8; ++i)
        out[i] = slot->value[i];
}

NO_INLINE void ethash_keccak256_64_be(ethash_w32 out[8], const ethash_w32* data)
{
    // The index mixes the words a mapping key and slot vary in: the last word of each half (the
    // low bytes of a number or an address), the first (a left-aligned bytesN key) and the fourth
    // (the high bytes of an address). On the corpus it hits as often as a mix of all 16 does.
    const uint32_t x = data[0] ^ data[3] ^ data[7] ^ data[15];
    const uint32_t index = (x * 0x9E3779B1u) >> (32 - KECCAK64_MEMO_BITS);
    struct keccak64_memo_slot* slot = &keccak64_memo[index];
    __asm__("" : "+r"(slot));  // Kept in a register: GCC would recompute it from the index.

    if (slot->value[0] != 0)
    {
        size_t i;
        for (i = 0; i < 16; ++i)
        {
            if (slot->key[i] != data[i])
                goto miss;
        }
        for (i = 0; i < 8; ++i)
            out[i] = slot->value[i];
        return;
    }
miss:
    keccak64_memo_fill(out, data, slot);
}
#if defined(__riscv) && __riscv_xlen == 32
/// The next word of the input from the non-determinism oracle (CSR 0x7C0), as the guest's reader
/// takes it.
static inline __attribute__((always_inline)) uint32_t read_input_word(void)
{
    uint32_t w;
    __asm__ volatile("csrrw %0, 0x7C0, x0" : "=r"(w)::"memory");
    return w;
}

/// Reads the ceil(@p size / 4) input words of a payload of @p size (more than 32) bytes into @p dst
/// as they arrive and absorbs them into the state of its Keccak-256 hash on the way, so that the
/// words are loaded once, not stored by the reader and loaded again by the hash. Returns 1 if the
/// hash equals the 8 words at @p key, which the caller must have stored apart from @p dst.
///
/// The words reach @p dst whole, whatever follows the payload in its last word, but only the
/// payload's bytes are absorbed: with those masked out, the unread bytes would let a key that
/// hashes the payload extended by one to three of them pass for the payload's own.
///
/// Keeps the state in buf[] as ethash_keccak256() does, so it must not run inside a hash.
NO_INLINE int ethash_keccak256_read_verify(ethash_w32* dst, size_t size, const ethash_w32* key)
{
    ethash_w32* const s = (ethash_w32*)buf;
    size_t i;
    size_t rem = size;
    size_t m;
    size_t r;
    uint32_t pad;
    if (rem >= 136)
    {
        // The state starts at zero: the first block is stored, not XORed in, and the rest of the
        // state zeroed.
#pragma GCC unroll 34
        for (i = 0; i < 34; ++i)
        {
            const uint32_t w = read_input_word();
            dst[i] = w;
            s[i] = w;
        }
#pragma GCC unroll 16
        for (; i < 50; ++i)
            s[i] = 0;
        keccak_permute_buf();
        dst += 34;
        rem -= 136;
        while (rem >= 136)
        {
#pragma GCC unroll 34
            for (i = 0; i < 34; ++i)
            {
                const uint32_t w = read_input_word();
                dst[i] = w;
                s[i] ^= w;
            }
            keccak_permute_buf();
            dst += 34;
            rem -= 136;
        }
        m = rem / 4;
        r = rem % 4;
        for (i = 0; i < m; ++i)
        {
            const uint32_t w = read_input_word();
            dst[i] = w;
            s[i] ^= w;
        }
        // The padding byte follows the payload's last byte, in the word that holds it.
        pad = (uint32_t)1 << (8 * r);
        if (r != 0)
        {
            const uint32_t w = read_input_word();
            dst[m] = w;
            pad |= w & (pad - 1);
        }
        s[m] ^= pad;
        s[33] ^= 0x80000000u;
    }
    else
    {
        m = rem / 4;
        r = rem % 4;
        for (i = 0; i < m; ++i)
        {
            const uint32_t w = read_input_word();
            dst[i] = w;
            s[i] = w;
        }
        pad = (uint32_t)1 << (8 * r);
        if (r != 0)
        {
            const uint32_t w = read_input_word();
            dst[m] = w;
            pad |= w & (pad - 1);
        }
        s[m] = pad;
        buf_zero_state_from(4 * (m + 1));
        s[33] |= 0x80000000u;
    }
    keccak_permute_buf();

    uint32_t diff = 0;
#pragma GCC unroll 8
    for (i = 0; i < 8; ++i)
        diff |= s[i] ^ key[i];
    return diff == 0;
}
#endif
#endif
