// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "baseline.hpp"
#include "instructions.hpp"
#include <bit>
#include <cstring>
#include <limits>
#include <memory>

namespace evmone::baseline
{
static_assert(std::is_move_constructible_v<CodeAnalysis>);
static_assert(std::is_move_assignable_v<CodeAnalysis>);
static_assert(!std::is_copy_constructible_v<CodeAnalysis>);
static_assert(!std::is_copy_assignable_v<CodeAnalysis>);
// A stateful deleter would grow CodeAnalysis past the 32-byte block it is shared in.
static_assert(sizeof(CodeStorage) == sizeof(uint8_t*));

size_t scan_jumpdests(JumpdestMap map, const uint8_t* code, size_t from, size_t limit) noexcept
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
    const auto* const base = reinterpret_cast<const int8_t*>(code);
    const auto* const end = base + limit;
    const auto* p = base + from;
#if EVMONE_JUMPDEST_BYTEMAP
    // Test 8 opcodes per bound check: a plain opcode then costs its load and one branch, where
    // the loop below spends 4 instructions on each. The reads run up to 7 bytes past the end,
    // into the zero padding of the code (see analyze_legacy()), and a 0 (STOP) is
    // neither PUSH nor JUMPDEST, so it only steps the walk past the end.
    // The mark for p + K lands at map + (p + K - base): with map - code fixed for the call, one
    // add forms the address and K folds into the store's offset. The difference is taken between
    // integers: code analyzed in place lies in another allocation than its map.
    const auto map_delta = reinterpret_cast<uintptr_t>(map) - reinterpret_cast<uintptr_t>(code);
    // Handle the PUSH or JUMPDEST..PUSH0 opcode op at p + K and step past it (and PUSH data).
    const auto special = [&]<std::ptrdiff_t K>(int8_t op) noexcept {
        if (op >= OP_PUSH1)
            p += std::ptrdiff_t{op} - OP_PUSH1 + 2 + K;
        else
        {
            // Most of these are JUMPDESTs: laid out inline, the mark shares the step with the
            // other opcodes instead of jumping back to it. Storing op (nonzero) saves a li.
            if (op == OP_JUMPDEST) [[likely]]
                reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(p) + map_delta)[K] =
                    static_cast<uint8_t>(op);
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
    while (p < end)
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
    return static_cast<size_t>(p - base);
}

namespace
{
#if EVMONE_IN_PLACE_CODE
/// The region of set_in_place_code_region(), empty while begin lies above every address.
struct InPlaceRegion
{
    uintptr_t begin = UINTPTR_MAX;
    uintptr_t end = 0;
};
// The guest runs one thread; a host test build may run VMs on several.
#if defined(AIRBENDER)
constinit InPlaceRegion in_place_region;
#else
constinit thread_local InPlaceRegion in_place_region;
#endif

/// Whether the code can be executed where it lies: inside the region and followed by 33 zero
/// bytes. The test of the bytes on every analysis keeps the copy as the fallback for anything the
/// host has not padded. The 9 aligned words read cover them with up to 3 bytes more on either
/// side, which the two shifts drop (the region's contract keeps those readable).
bool is_padded_in_place(bytes_view code) noexcept
{
    static_assert(std::endian::native == std::endian::little);
    typedef uint32_t __attribute__((may_alias)) word;
    const auto begin = reinterpret_cast<uintptr_t>(code.data());
    const auto end = begin + code.size();
    if (begin < in_place_region.begin || end > in_place_region.end)
        return false;
    const word* const w = reinterpret_cast<const word*>(end & ~uintptr_t{3});
    const auto sh = static_cast<unsigned>(end & 3) * 8;
    uint32_t acc = (w[0] >> sh) | (w[8] << (24 - sh));
    for (size_t i = 1; i < 8; ++i)
        acc |= w[i];
    return acc == 0;
}
#endif

CodeAnalysis analyze_legacy(bytes_view code)
{
    // We need at most 33 bytes of code padding: 32 for possible missing all data bytes of
    // the PUSH32 at the code end; and one more byte for STOP to guarantee there is a terminating
    // instruction at the code end.
    static constexpr auto PADDING = 32 + 1;

    static constexpr auto BITSET_ALIGNMENT = alignof(BitsetSpan::word_type);

    const auto padded_code_size = code.size() + PADDING;
    [[maybe_unused]] const auto bitset_words = (code.size() + (BitsetSpan::WORD_BITS)) / BitsetSpan::WORD_BITS;
#if EVMONE_IN_PLACE_CODE
    // Code in place is scanned on demand, never up front, into a byte map of its own.
    static_assert(EVMONE_LAZY_JUMPDESTS && EVMONE_JUMPDEST_BYTEMAP);
#endif
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // The copy starts at the code's own offset modulo 32: the guest's memcpy then copies the
    // bulk in 32-byte CSR chunks (0.125 instructions a byte) instead of word by word (0.5), and
    // 2.8 MB of code is copied per block. The byte map follows the padded code word-aligned.
    //
    // calloc hands out fresh heap memory that is already zero, so neither the padding nor the
    // map is cleared: a CSR memset of the map alone would add 17.8M BigInt calls to a corpus.
    // The size is the bitset layout's plus a multiple of 32, which leaves every later
    // allocation at its address modulo 32 (the memcpy CSR path depends on it).
    const auto head_size =
        32 + padded_code_size + BITSET_ALIGNMENT + bitset_words * sizeof(BitsetSpan::word_type);
    const auto map_size = code.size() + 8;
    const auto map_alloc_size = (map_size + 6 + 31) & ~size_t{31};
    if (is_padded_in_place(code))
    {
        // Only the map is stored (2.8 MB of code a block is not copied). The size is the copy's
        // less a multiple of 32, which leaves later allocations at their addresses modulo 32 too.
        CodeStorage storage{
            static_cast<uint8_t*>(std::calloc(1, (head_size & 31) + map_alloc_size))};
        uint8_t* const map = storage.get();
        return {std::move(storage), code.data(), code.size(), map};
    }
    CodeStorage storage{static_cast<uint8_t*>(std::calloc(1, head_size + map_alloc_size))};
    const auto base = reinterpret_cast<uintptr_t>(storage.get());
    const auto code_off = (reinterpret_cast<uintptr_t>(code.data()) - base) & 31;
    uint8_t* const padded = storage.get() + code_off;
    const auto map_off = (code_off + padded_code_size + 3) / 4 * 4;
    uint8_t* const map = storage.get() + map_off;
    std::memcpy(padded, code.data(), code.size());
#if !EVMONE_LAZY_JUMPDESTS
    // Scan the padded copy: a truncated PUSH at the end advances past the last opcode by up to
    // 32 bytes, which stays inside this allocation but not inside the caller's.
    scan_jumpdests(map, padded, 0, code.size());
#endif
    return {std::move(storage), padded, code.size(), map};
#else
#if EVMONE_IN_PLACE_CODE
    if (is_padded_in_place(code))
    {
        // Only the map is stored, zeroed here: host memory is not fresh.
        const auto map_size = code.size() + 8;
        auto storage = std::make_unique_for_overwrite<uint8_t[]>(map_size);
        std::fill_n(storage.get(), map_size, 0);
        uint8_t* const map = storage.get();
        return {std::move(storage), code.data(), code.size(), map};
    }
#endif
    const auto aligned_code_size =
        (padded_code_size + (BITSET_ALIGNMENT - 1)) / BITSET_ALIGNMENT * BITSET_ALIGNMENT;
#if EVMONE_JUMPDEST_BYTEMAP
    // The map follows the padded code in the same storage, as on the guest. Host memory is not
    // fresh, so the fill below zeroes the map along with the padding.
    const auto total_size = aligned_code_size + code.size() + 8;
#else
    const auto total_size = aligned_code_size + bitset_words * sizeof(BitsetSpan::word_type);
#endif

    auto storage = std::make_unique_for_overwrite<uint8_t[]>(total_size);
#if defined(AIRBENDER) && defined(__riscv)
    // Use explicit memcpy/memset — std::ranges::copy and std::fill_n may not dispatch
    // to our optimized builtins on bare-metal toolchains (riscv-none-elf-gcc).
    std::memcpy(storage.get(), code.data(), code.size());
    std::memset(&storage[code.size()], 0, total_size - code.size());
#else
    std::ranges::copy(code, storage.get());                           // Copy code.
    std::fill_n(&storage[code.size()], total_size - code.size(), 0);  // Pad code and init the jumpdest map.
#endif

#if EVMONE_JUMPDEST_BYTEMAP
    const JumpdestMap jumpdest_map = &storage[aligned_code_size];
#else
    const auto bitset_storage =
        new (&storage[aligned_code_size]) BitsetSpan::word_type[bitset_words];
    const JumpdestMap jumpdest_map{bitset_storage};
#endif
#if !EVMONE_LAZY_JUMPDESTS
    // Scan the padded copy: a truncated PUSH at the end advances past the last opcode by up to
    // 32 bytes, which stays inside this allocation but not inside the caller's.
    scan_jumpdests(jumpdest_map, storage.get(), 0, code.size());
#endif

    return {std::move(storage), code.size(), jumpdest_map};
#endif
}
}  // namespace

CodeAnalysis analyze(bytes_view code)
{
    return analyze_legacy(code);
}

#if EVMONE_IN_PLACE_CODE
void set_in_place_code_region(const uint8_t* begin, const uint8_t* end) noexcept
{
    in_place_region = begin != nullptr ?
                          InPlaceRegion{reinterpret_cast<uintptr_t>(begin),
                              reinterpret_cast<uintptr_t>(end)} :
                          InPlaceRegion{};
}
#endif
}  // namespace evmone::baseline
