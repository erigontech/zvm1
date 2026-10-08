/* ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
 * Copyright 2026 The zvm1 Authors (modifications)
 * Copyright 2018-2019 Pawel Bylica (original)
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "hash_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifndef __cplusplus
#define noexcept  // Ignore noexcept in C code.
#endif

#ifdef __cplusplus
extern "C" {
#endif

union ethash_hash256 ethash_keccak256(const uint8_t* data, size_t size) noexcept;
union ethash_hash256 ethash_keccak256_32(const uint8_t data[32]) noexcept;

/// Whether ethash_keccak256() of the @p size bytes at @p data is the 32 bytes at @p expected, both
/// at any alignment. On the Airbender guest the hash stays in the Keccak state and is compared
/// there, which spares a caller that only checks a hash its stores and reloads.
bool ethash_keccak256_eq(const uint8_t* expected, const uint8_t* data, size_t size) noexcept;

/// ethash_keccak256() of the @p size bytes at @p data, which must be 8-byte aligned and hold at least
/// @p blocks (1 or more) whole 136-byte blocks, that also leaves the Keccak state after those
/// blocks in @p slot (32 lanes, 256-byte aligned): the state of the first 136 * @p blocks bytes.
union ethash_hash256 ethash_keccak256_snap(
    const uint8_t* data, size_t size, size_t blocks, uint64_t* slot) noexcept;

/// ethash_keccak256() of the @p size bytes at @p data (8-byte aligned) given the state ethash_keccak256_snap()
/// left in @p slot for the same @p blocks and the same first 136 * @p blocks bytes of data: absorbs only
/// the rest. The slot is permuted in place and so holds no snapshot afterwards.
union ethash_hash256 ethash_keccak256_resume(
    uint64_t* slot, size_t blocks, const uint8_t* data, size_t size) noexcept;

/// ethash_keccak256() of a full trie branch with some of its hash references replaced, read from
/// the branch itself rather than from a patched copy of it. @p node is the 532-byte branch, 8-byte
/// aligned: a list header f9 02 11, 16 slots of 0xa0 and a 32-byte hash (slot j's hash at byte
/// 4 + 33 j) and the empty value 0x80. The hashed bytes are those with the hash of each slot j in
/// @p dirty (bit j) replaced by the 32 bytes at @p ptrs[j] (any alignment) if that is not null,
/// else by the 32 bytes at @p hashes + 32 j (8-byte aligned).
/// With @p blocks nonzero, @p slot holds the state ethash_keccak256_snap() left for the node and
/// that many blocks, which must not hold any byte of a dirty slot, and is consumed as by
/// ethash_keccak256_resume(); with @p blocks 0, @p slot is not used.
union ethash_hash256 ethash_keccak256_full_branch(uint64_t* slot, size_t blocks,
    const uint8_t* node, uint32_t dirty, const uint8_t* hashes, const uint8_t* const* ptrs) noexcept;

#if defined(AIRBENDER)
/// A 32-bit word of a byte buffer or of a uint256 (stored as 64-bit words): may alias them.
typedef uint32_t __attribute__((may_alias)) ethash_w32;

/// Keccak-256 of the 64 bytes at @p data, which must be 4-byte aligned, memoized. The bytes are EVM
/// memory in the word layout (see evmone/word_layout.hpp): each word holds the big-endian number of
/// its 4 bytes. Stores the hash
/// to @p out as the big-endian number, in little-endian words (the uint256 KECCAK256 pushes).
void ethash_keccak256_64_be(ethash_w32 out[8], const ethash_w32* data) noexcept;

#if defined(__riscv) && __riscv_xlen == 32
/// Reads the ceil(@p size / 4) words of a payload of @p size bytes (more than 32) from the guest's
/// input into @p dst, hashing it as they arrive: 1 if its Keccak-256 equals the 8 words at @p key.
/// @p dst and @p key must not overlap.
int ethash_keccak256_read_verify(ethash_w32* dst, size_t size, const ethash_w32* key) noexcept;
#endif
#endif

#ifdef __cplusplus
}
#endif
