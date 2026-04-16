// ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
// Copyright 2018 Pawel Bylica.
// SPDX-License-Identifier: Apache-2.0

#include "keccak.h"

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
static const uint64_t __attribute__((aligned(32))) keccak_zeros[4] = {0, 0, 0, 0};

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

/// Keccak-f[1600] via airbender CSR 0x7CB delegation.
/// 649 consecutive CSR writes — the transpiler's preprocess_bytecode
/// scans for exactly 649 contiguous csrrw instructions.
static void syscall_keccak_permute(uint64_t state[25])
{
    int i;
    // Zero buf[25..31] before copying state. 48 bytes = 6 uint64_t.
    // Use scalar stores (simpler than CSR for this small range).
    for (i = 25; i < 31; i++)
        buf[i] = 0;
    for (i = 0; i < 25; i++)
        buf[i] = state[i];

    register uint32_t ctrl __asm__("x10") = 0;
    register void*    sptr __asm__("x11") = (void*)buf;
    __asm__ __volatile__(
        ".rept 649\n"
        "  csrrw x0, 0x7CB, x0\n"
        ".endr\n"
        : "+r"(ctrl)
        : "r"(sptr)
        : "memory"
    );
    for (i = 0; i < 25; i++)
        state[i] = buf[i];
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

#if !__has_builtin(__builtin_memcpy) && !defined(__GNUC__)
#include <string.h>
#define __builtin_memcpy memcpy
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define to_le64(X) __builtin_bswap64(X)
#else
#define to_le64(X) X
#endif

/// Loads 64-bit integer from given memory location as little-endian number.
static inline ALWAYS_INLINE uint64_t load_le(const uint8_t* data)
{
#if defined(__riscv) && __riscv_xlen == 32
    // On rv32im with -mstrict-align, __builtin_memcpy generates byte-by-byte loads
    // (~16 insns for 8 bytes). Use word loads when 4-byte aligned (~4 insns).
    // RISC-V is little-endian so to_le64 is a no-op.
    if (__builtin_expect(((uintptr_t)data & 3) == 0, 1))
    {
        const uint32_t* w = (const uint32_t*)data;
        return (uint64_t)w[0] | ((uint64_t)w[1] << 32);
    }
#endif
    uint64_t word;
    __builtin_memcpy(&word, data, sizeof(word));
    return to_le64(word);
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


static inline ALWAYS_INLINE void keccak(
    uint64_t* out, size_t bits, const uint8_t* data, size_t size)
{
    static const size_t word_size = sizeof(uint64_t);
    const size_t hash_size = bits / 8;
    const size_t block_size = (1600 - bits * 2) / 8;

    size_t i;
    uint64_t last_word = 0;
    uint8_t* last_word_iter = (uint8_t*)&last_word;

#if defined(AIRBENDER)
    // Work directly on the static CSR-aligned buf[] to avoid the
    // state→buf→state copies that syscall_keccak_permute would do
    // on every permutation call (saves 50 word copies per call).
    buf_zero_all();

    while (size >= block_size)
    {
        for (i = 0; i < (block_size / word_size); ++i)
        {
            buf[i] ^= load_le(data);
            data += word_size;
        }

        {
            register uint32_t ctrl __asm__("x10") = 0;
            register void*    sptr __asm__("x11") = (void*)buf;
            __asm__ __volatile__(
                ".rept 649\n"
                "  csrrw x0, 0x7CB, x0\n"
                ".endr\n"
                : "+r"(ctrl)
                : "r"(sptr)
                : "memory"
            );
        }

        size -= block_size;
    }

    {
        uint64_t* buf_iter = buf;
        while (size >= word_size)
        {
            *buf_iter ^= load_le(data);
            ++buf_iter;
            data += word_size;
            size -= word_size;
        }

        while (size > 0)
        {
            *last_word_iter = *data;
            ++last_word_iter;
            ++data;
            --size;
        }
        *last_word_iter = 0x01;
        *buf_iter ^= to_le64(last_word);
    }

    buf[(block_size / word_size) - 1] ^= 0x8000000000000000;

    {
        register uint32_t ctrl __asm__("x10") = 0;
        register void*    sptr __asm__("x11") = (void*)buf;
        __asm__ __volatile__(
            ".rept 649\n"
            "  csrrw x0, 0x7CB, x0\n"
            ".endr\n"
            : "+r"(ctrl)
            : "r"(sptr)
            : "memory"
        );
    }

    for (i = 0; i < (hash_size / word_size); ++i)
        out[i] = to_le64(buf[i]);
#else
    uint64_t state[25] = {0};
    uint64_t* state_iter;

    while (size >= block_size)
    {
        for (i = 0; i < (block_size / word_size); ++i)
        {
            state[i] ^= load_le(data);
            data += word_size;
        }

        keccakf1600_best(state);

        size -= block_size;
    }

    state_iter = state;

    while (size >= word_size)
    {
        *state_iter ^= load_le(data);
        ++state_iter;
        data += word_size;
        size -= word_size;
    }

    while (size > 0)
    {
        *last_word_iter = *data;
        ++last_word_iter;
        ++data;
        --size;
    }
    *last_word_iter = 0x01;
    *state_iter ^= to_le64(last_word);

    state[(block_size / word_size) - 1] ^= 0x8000000000000000;

    keccakf1600_best(state);

    for (i = 0; i < (hash_size / word_size); ++i)
        out[i] = to_le64(state[i]);
#endif
}

union ethash_hash256 ethash_keccak256(const uint8_t* data, size_t size)
{
    union ethash_hash256 hash;
    // For keccak-256: block_size = (1600 - 256*2) / 8 = 136 bytes.
    // Most EVM inputs are < 136 bytes (single block). Specialize.
    if (size < 136)
    {
#if defined(AIRBENDER)
        // Write directly to the static CSR-aligned buf — avoid state→buf→state copies.
        {
            int i;
            buf_zero_all();

            uint64_t* buf_iter = buf;
            const uint8_t* d = data;
            size_t remaining = size;
            while (remaining >= 8)
            {
                *buf_iter++ ^= load_le(d);
                d += 8;
                remaining -= 8;
            }
            uint64_t last_word = 0;
            uint8_t* lw = (uint8_t*)&last_word;
            for (i = 0; i < (int)remaining; ++i)
                lw[i] = d[i];
            lw[remaining] = 0x01;
            *buf_iter ^= to_le64(last_word);
            buf[16] ^= 0x8000000000000000ULL;

            register uint32_t ctrl __asm__("x10") = 0;
            register void*    sptr __asm__("x11") = (void*)buf;
            __asm__ __volatile__(
                ".rept 649\n"
                "  csrrw x0, 0x7CB, x0\n"
                ".endr\n"
                : "+r"(ctrl)
                : "r"(sptr)
                : "memory"
            );
            hash.word64s[0] = to_le64(buf[0]);
            hash.word64s[1] = to_le64(buf[1]);
            hash.word64s[2] = to_le64(buf[2]);
            hash.word64s[3] = to_le64(buf[3]);
            return hash;
        }
#else
        size_t i;
        uint64_t state[25] = {0};
        uint64_t* state_iter = state;
        const uint8_t* d = data;
        size_t remaining = size;

        while (remaining >= 8)
        {
            *state_iter++ ^= load_le(d);
            d += 8;
            remaining -= 8;
        }

        // Handle remaining bytes + padding byte 0x01
        uint64_t last_word = 0;
        uint8_t* lw = (uint8_t*)&last_word;
        for (i = 0; i < remaining; ++i)
            lw[i] = d[i];
        lw[remaining] = 0x01;
        *state_iter ^= to_le64(last_word);

        state[16] ^= 0x8000000000000000ULL;  // block_size/8 - 1 = 16

        keccakf1600_best(state);

        hash.word64s[0] = to_le64(state[0]);
        hash.word64s[1] = to_le64(state[1]);
        hash.word64s[2] = to_le64(state[2]);
        hash.word64s[3] = to_le64(state[3]);
        return hash;
#endif
    }
    keccak(hash.word64s, 256, data, size);
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
        buf[0] = load_le(data);
        buf[1] = load_le(data + 8);
        buf[2] = load_le(data + 16);
        buf[3] = load_le(data + 24);
        buf[4] = 0x0000000000000001ULL;
        buf[16] = 0x8000000000000000ULL;

        register uint32_t ctrl __asm__("x10") = 0;
        register void*    sptr __asm__("x11") = (void*)buf;
        __asm__ __volatile__(
            ".rept 649\n"
            "  csrrw x0, 0x7CB, x0\n"
            ".endr\n"
            : "+r"(ctrl)
            : "r"(sptr)
            : "memory"
        );
        hash.word64s[0] = to_le64(buf[0]);
        hash.word64s[1] = to_le64(buf[1]);
        hash.word64s[2] = to_le64(buf[2]);
        hash.word64s[3] = to_le64(buf[3]);
        return hash;
    }
#else
    // Specialized path: 32 bytes input, keccak-256.
    // block_size = (1600 - 256*2) / 8 = 136 bytes.
    // 32 < 136, so no multi-block processing needed.
    // 32 / 8 = 4 full words, 0 remaining bytes.
    uint64_t state[25] = {0};
    state[0] = load_le(data);
    state[1] = load_le(data + 8);
    state[2] = load_le(data + 16);
    state[3] = load_le(data + 24);
    // Padding: 0x01 byte after data, 0x80 at end of block.
    state[4] = 0x0000000000000001ULL;   // to_le64(0x01) at position 32
    state[16] ^= 0x8000000000000000ULL; // block_size/8 - 1 = 16
    keccakf1600_best(state);
    hash.word64s[0] = to_le64(state[0]);
    hash.word64s[1] = to_le64(state[1]);
    hash.word64s[2] = to_le64(state[2]);
    hash.word64s[3] = to_le64(state[3]);
    return hash;
#endif
}
