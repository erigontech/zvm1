// ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2018-2019 Pawel Bylica (original)
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "keccak.h"

namespace ethash
{
inline hash256 keccak256(const uint8_t* data, size_t size) noexcept
{
    return ethash_keccak256(data, size);
}

inline hash256 keccak256(const hash256& input) noexcept
{
    return ethash_keccak256_32(input.bytes);
}

static constexpr auto keccak256_32 = ethash_keccak256_32;

}  // namespace ethash
