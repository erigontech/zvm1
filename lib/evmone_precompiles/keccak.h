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

#ifdef __cplusplus
}
#endif
