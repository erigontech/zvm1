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

#if defined(AIRBENDER)
/// A 32-bit word of a byte buffer or of a uint256 (stored as 64-bit words): may alias them.
typedef uint32_t __attribute__((may_alias)) ethash_w32;

/// Keccak-256 of the 64 bytes at @p data, which must be 4-byte aligned, memoized: stores the hash
/// to @p out as the big-endian number, in little-endian words (the uint256 KECCAK256 pushes).
void ethash_keccak256_64_be(ethash_w32 out[8], const ethash_w32* data) noexcept;
#endif

#ifdef __cplusplus
}
#endif
