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
#if defined(SP1TURBO) || defined(SP1)
#define DEFAULT_keccakf1600 syscall_keccak_permute
#else
#define DEFAULT_keccakf1600 keccakf1600_generic
#endif

static void (*keccakf1600_best)(uint64_t[25]) = DEFAULT_keccakf1600;


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

#if KECCAK_INLINE_STATE_CLEAR
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
    keccak(hash.word64s, 256, data, size);
    return hash;
}

union ethash_hash256 ethash_keccak256_32(const uint8_t data[32])
{
    union ethash_hash256 hash;
    keccak(hash.word64s, 256, data, 32);
    return hash;
}
