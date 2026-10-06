/* ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
 * Copyright 2026 The zvm1 Authors (modifications)
 * Copyright 2018-2019 Pawel Bylica (original)
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "hash_types.h"
#include <stddef.h>

#ifndef __cplusplus
#define noexcept  // Ignore noexcept in C code.
#endif

#ifdef __cplusplus
extern "C" {
#endif

union ethash_hash256 ethash_keccak256(const uint8_t* data, size_t size) noexcept;
union ethash_hash256 ethash_keccak256_32(const uint8_t data[32]) noexcept;

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

#if defined(AIRBENDER)
/// A 32-bit word of a byte buffer or of a uint256 (stored as 64-bit words): may alias them.
typedef uint32_t __attribute__((may_alias)) ethash_w32;

/// Keccak-256 of the 64 bytes at @p data, which must be 4-byte aligned, memoized: stores the hash
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
