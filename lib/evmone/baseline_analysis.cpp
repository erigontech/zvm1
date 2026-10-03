// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "baseline.hpp"
#include "instructions.hpp"
#include <cstring>
#include <limits>
#include <memory>

namespace evmone::baseline
{
static_assert(std::is_move_constructible_v<CodeAnalysis>);
static_assert(std::is_move_assignable_v<CodeAnalysis>);
static_assert(!std::is_copy_constructible_v<CodeAnalysis>);
static_assert(!std::is_copy_assignable_v<CodeAnalysis>);

namespace
{
void analyze_jumpdests(BitsetSpan map, bytes_view code) noexcept
{
    // To find if op is any PUSH opcode (OP_PUSH1 <= op <= OP_PUSH32)
    // it can be noticed that OP_PUSH32 is INT8_MAX (0x7f) therefore,
    // op <= OP_PUSH32 is always true for a signed byte and can be skipped.
    static_assert(OP_PUSH32 == std::numeric_limits<int8_t>::max());

    // Walk signed bytes: the PUSH test's sign extension folds into the load and opcodes >= 0x80
    // turn negative, so one `op < OP_JUMPDEST` check rejects all but PUSH and JUMPDEST.
    //
    // On rv64im with GCC this makes the common path 4 instructions instead of 9, and there an
    // executed instruction is a proven cycle. clang needs one more: llvm/llvm-project#217273.
    const auto* const base = reinterpret_cast<const int8_t*>(code.data());
    const auto* const end = base + code.size();
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // Test 8 opcodes per bound check: a plain opcode then costs its load and one branch, where
    // the loop above spends 4 instructions on each. The reads run up to 7 bytes past the end,
    // into the zero padding of the analysis copy (see analyze_legacy()), and a 0 (STOP) is
    // neither PUSH nor JUMPDEST, so it only steps the walk past the end.
    const auto* p = base;
    // Handle the PUSH or JUMPDEST..PUSH0 opcode op at p + K and step past it (and PUSH data).
    const auto special = [&]<std::ptrdiff_t K>(int8_t op) noexcept {
        if (op >= OP_PUSH1)
            p += std::ptrdiff_t{op} - OP_PUSH1 + 2 + K;
        else
        {
            if (op == OP_JUMPDEST)
                map.set(static_cast<size_t>(p + K - base));
            p += K + 1;
        }
    };
    while (p < end)
    {
        if (const auto op = p[0]; op >= OP_JUMPDEST)
            special.operator()<0>(op);
        else if (const auto op1 = p[1]; op1 >= OP_JUMPDEST)
            special.operator()<1>(op1);
        else if (const auto op2 = p[2]; op2 >= OP_JUMPDEST)
            special.operator()<2>(op2);
        else if (const auto op3 = p[3]; op3 >= OP_JUMPDEST)
            special.operator()<3>(op3);
        else if (const auto op4 = p[4]; op4 >= OP_JUMPDEST)
            special.operator()<4>(op4);
        else if (const auto op5 = p[5]; op5 >= OP_JUMPDEST)
            special.operator()<5>(op5);
        else if (const auto op6 = p[6]; op6 >= OP_JUMPDEST)
            special.operator()<6>(op6);
        else if (const auto op7 = p[7]; op7 >= OP_JUMPDEST)
            special.operator()<7>(op7);
        else
            p += 8;
    }
#else
    for (const auto* p = base; p < end;)
    {
        const auto op = *p;
        if (op >= OP_JUMPDEST)  // Everything below is neither, including every opcode >= 0x80.
        {
            if (op >= OP_PUSH1)  // If any PUSH opcode (see explanation above).
            {
                // Widen before subtracting: left in int, clang narrows the advance back to
                // i8 and masks it, which lengthens the loop. llvm/llvm-project#217314
                p += std::ptrdiff_t{op} - OP_PUSH1 + 2;  // Skip the opcode and its PUSH data.
                continue;
            }
            if (op == OP_JUMPDEST) [[unlikely]]
                map.set(static_cast<size_t>(p - base));
        }
        ++p;
    }
#endif
}

CodeAnalysis analyze_legacy(bytes_view code)
{
    // We need at most 33 bytes of code padding: 32 for possible missing all data bytes of
    // the PUSH32 at the code end; and one more byte for STOP to guarantee there is a terminating
    // instruction at the code end.
    static constexpr auto PADDING = 32 + 1;

    static constexpr auto BITSET_ALIGNMENT = alignof(BitsetSpan::word_type);

    const auto padded_code_size = code.size() + PADDING;
    const auto bitset_words = (code.size() + (BitsetSpan::WORD_BITS)) / BitsetSpan::WORD_BITS;
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // The copy starts at the code's own offset modulo 32: the guest's memcpy then copies the
    // bulk in 32-byte CSR chunks (0.125 instructions a byte) instead of word by word (0.5), and
    // 2.8 MB of code is copied per block. The bitset follows the padded code word-aligned.
    auto storage = std::make_unique_for_overwrite<uint8_t[]>(
        32 + padded_code_size + BITSET_ALIGNMENT + bitset_words * sizeof(BitsetSpan::word_type));
    const auto base = reinterpret_cast<uintptr_t>(storage.get());
    const auto code_off = (reinterpret_cast<uintptr_t>(code.data()) - base) & 31;
    uint8_t* const padded = storage.get() + code_off;
    const auto bitset_off =
        ((code_off + padded_code_size + (BITSET_ALIGNMENT - 1)) / BITSET_ALIGNMENT) *
        BITSET_ALIGNMENT;
    const auto total_size = bitset_off + bitset_words * sizeof(BitsetSpan::word_type);
    std::memcpy(padded, code.data(), code.size());
    std::memset(padded + code.size(), 0, total_size - code_off - code.size());
    const auto bitset_storage =
        new (&storage[bitset_off]) BitsetSpan::word_type[bitset_words];
    const BitsetSpan jumpdest_bitset{bitset_storage};
    // Scan the padded copy: a truncated PUSH at the end advances past the last opcode by up to
    // 32 bytes, which stays inside this allocation but not inside the caller's.
    analyze_jumpdests(jumpdest_bitset, {padded, code.size()});
    return {std::move(storage), padded, code.size(), jumpdest_bitset};
#else
    const auto aligned_code_size =
        (padded_code_size + (BITSET_ALIGNMENT - 1)) / BITSET_ALIGNMENT * BITSET_ALIGNMENT;
    const auto total_size = aligned_code_size + bitset_words * sizeof(BitsetSpan::word_type);

    auto storage = std::make_unique_for_overwrite<uint8_t[]>(total_size);
#if defined(AIRBENDER) && defined(__riscv)
    // Use explicit memcpy/memset — std::ranges::copy and std::fill_n may not dispatch
    // to our optimized builtins on bare-metal toolchains (riscv-none-elf-gcc).
    std::memcpy(storage.get(), code.data(), code.size());
    std::memset(&storage[code.size()], 0, total_size - code.size());
#else
    std::ranges::copy(code, storage.get());                           // Copy code.
    std::fill_n(&storage[code.size()], total_size - code.size(), 0);  // Pad code and init bitset.
#endif

    const auto bitset_storage =
        new (&storage[aligned_code_size]) BitsetSpan::word_type[bitset_words];
    const BitsetSpan jumpdest_bitset{bitset_storage};
    // Scan the padded copy: a truncated PUSH at the end advances past the last opcode by up to
    // 32 bytes, which stays inside this allocation but not inside the caller's.
    analyze_jumpdests(jumpdest_bitset, {storage.get(), code.size()});

    return {std::move(storage), code.size(), jumpdest_bitset};
#endif
}
}  // namespace

CodeAnalysis analyze(bytes_view code)
{
    return analyze_legacy(code);
}
}  // namespace evmone::baseline
