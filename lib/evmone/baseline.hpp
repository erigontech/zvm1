// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <evmc/utils.h>
#include <algorithm>
#include <cstdlib>
#include <memory>

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(EVMONE_RV32_DISPATCH_TEST)
/// JUMPDEST analysis on demand: the scan runs up to the highest jump target checked so far
/// instead of over the whole code up front (10.7% of the analyzed bytes on mainnet lie past
/// every target). EVMONE_RV32_DISPATCH_TEST builds it on the host for testing.
#define EVMONE_LAZY_JUMPDESTS 1
/// The JUMPDESTs go in a byte map (one byte per code position, nonzero at a JUMPDEST) instead of
/// a bit set: the scan marks with one store and a jump tests with one byte load.
#define EVMONE_JUMPDEST_BYTEMAP 1
/// Code the host keeps padded in place is analyzed without a copy, see set_in_place_code_region().
#define EVMONE_IN_PLACE_CODE 1
#else
#define EVMONE_LAZY_JUMPDESTS 0
#define EVMONE_JUMPDEST_BYTEMAP 0
#define EVMONE_IN_PLACE_CODE 0
#endif

namespace evmone
{
using evmc::bytes_view;
class ExecutionState;
class VM;

/// A span type for a bitset.
struct BitsetSpan
{
    // On rv32im, 64-bit shift/mask/load in test() costs ~6-8 instructions per JUMP/JUMPI.
    // Using 32-bit words makes each bitset access a single-instruction shift, AND, and load.
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    using word_type = uint32_t;
#else
    using word_type = uint64_t;
#endif
    static constexpr size_t WORD_BITS = sizeof(word_type) * 8;

    word_type* m_array = nullptr;

    explicit BitsetSpan(word_type* array) noexcept : m_array{array} {}

    [[nodiscard]] bool test(size_t index) const noexcept
    {
        // Shift the word down to the bit instead of materializing a mask to test against.
        return ((m_array[index / WORD_BITS] >> (index % WORD_BITS)) & 1) != 0;
    }

    void set(size_t index) const noexcept
    {
        const auto& [word, bit_mask] = get_ref(index);
        word |= bit_mask;
    }

private:
    struct Ref
    {
        word_type& word_ref;
        word_type bit_mask;
    };

    [[nodiscard, gnu::always_inline]] Ref get_ref(size_t index) const noexcept
    {
        const auto word_index = index / WORD_BITS;
        const auto bit_index = index % WORD_BITS;
        const auto bit_mask = word_type{1} << bit_index;
        return {m_array[word_index], bit_mask};
    }
};

namespace baseline
{
#if EVMONE_JUMPDEST_BYTEMAP
/// One byte per code position, nonzero at a JUMPDEST. The map must start zeroed. The scan finds a
/// mark's address by an integer offset from the code's address, not by pointer arithmetic: the
/// map can lie in another allocation than the code (see set_in_place_code_region()).
using JumpdestMap = uint8_t*;
#else
using JumpdestMap = BitsetSpan;
#endif

#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
/// The analysis storage comes from calloc: the guest's calloc hands out fresh heap memory
/// without clearing it again (see simple_allocator.cpp).
struct FreeDeleter
{
    void operator()(uint8_t* p) const noexcept { std::free(p); }
};
using CodeStorage = std::unique_ptr<uint8_t[], FreeDeleter>;
#else
using CodeStorage = std::unique_ptr<uint8_t[]>;
#endif

/// Classifies the code positions in [from, limit) (and the ones a PUSH's data carries the scan
/// past), setting the JUMPDEST marks in map, and returns the first position left unclassified.
/// The code must be the padded copy: the scan reads up to 7 bytes past limit. With limit at the
/// code size this is the whole analysis.
EVMC_EXPORT size_t scan_jumpdests(
    JumpdestMap map, const uint8_t* code, size_t from, size_t limit) noexcept;

#if EVMONE_IN_PLACE_CODE
/// Lets analyze() use legacy code lying in [begin, end) where it is, instead of copying it to
/// append the 33 zero bytes of padding the interpreter needs (a STOP after the last opcode and the
/// data of a PUSH32 cut off by the end): it does so for code followed by 33 zero bytes, which it
/// checks each time. The caller guarantees that no byte in [begin, end + 36) changes while an
/// analysis of such code lives, and resets the region (nullptr, nullptr) before that stops
/// holding. One region per thread; a new one replaces the last.
EVMC_EXPORT void set_in_place_code_region(const uint8_t* begin, const uint8_t* end) noexcept;
#endif

class CodeAnalysis
{
private:
    bytes_view m_code;  ///< The executable code.

    /// Storage for the padded code for faster legacy code execution, and the JUMPDEST map.
    /// m_code points into it, or to code analyzed in place (see set_in_place_code_region()).
    CodeStorage m_padded_code;

    JumpdestMap m_jumpdest_map{nullptr};
#if EVMONE_LAZY_JUMPDESTS
    /// Positions below this are classified in the map. Never above the code size.
    mutable size_t m_scanned = 0;
#endif

public:
    /// Constructor for legacy code.
    CodeAnalysis(CodeStorage padded_code, size_t code_size, JumpdestMap map)
      : m_code{padded_code.get(), code_size},
        m_padded_code{std::move(padded_code)},
        m_jumpdest_map{map}
    {}

    /// Constructor for legacy code whose padded copy starts inside the owned storage, or which
    /// is padded where it lies.
    CodeAnalysis(CodeStorage storage, const uint8_t* padded_code, size_t code_size,
        JumpdestMap map)
      : m_code{padded_code, code_size}, m_padded_code{std::move(storage)}, m_jumpdest_map{map}
    {}

    /// The executable code. This is where the interpreter should start execution.
    [[nodiscard]] bytes_view code() const noexcept { return m_code; }

    /// Check if given position is valid jump destination. Use only for legacy code.
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // On rv32im, callers already verified position fits in 32 bits.
    // Use uint32_t to avoid 64-bit comparison with size().
    [[nodiscard]] bool check_jumpdest(uint32_t position) const noexcept
#else
    [[nodiscard]] bool check_jumpdest(uint64_t position) const noexcept
#endif
    {
#if EVMONE_LAZY_JUMPDESTS
        // m_scanned never exceeds the code size, so a position below it is both inside the code
        // and classified: one comparison covers the common case.
        if (position >= m_scanned) [[unlikely]]
        {
            if (position >= m_code.size())
                return false;
            scan_to(static_cast<size_t>(position));
        }
#else
        if (position >= m_code.size())
            return false;
#endif
#if EVMONE_JUMPDEST_BYTEMAP
        return m_jumpdest_map[position] != 0;
#else
        return m_jumpdest_map.test(static_cast<size_t>(position));
#endif
    }

#if EVMONE_JUMPDEST_BYTEMAP
    /// The JUMPDEST map, for a caller that keeps it in a register: see check_jumpdest_in().
    [[nodiscard]] JumpdestMap jumpdest_map() const noexcept { return m_jumpdest_map; }

    /// check_jumpdest() with the map passed in, as jumpdest_map() returned it, which saves its
    /// load.
    [[nodiscard]] bool check_jumpdest_in(JumpdestMap map, uint32_t position) const noexcept
    {
        // The scan falls through to the one map test: returning check_jumpdest()'s result instead
        // makes GCC merge the two answers into a register and test that on every jump.
        if (position >= m_scanned) [[unlikely]]
        {
            if (position >= m_code.size())
                return false;
            scan_to(position);
        }
        return map[position] != 0;
    }
#endif

private:
#if EVMONE_LAZY_JUMPDESTS
    /// Classifies the positions up to and including position, which is inside the code.
    /// Out of line, so that the scan loop does not share registers with the interpreter's hot
    /// state in dispatch_cgoto().
    [[gnu::noinline]] void scan_to(size_t position) const noexcept
    {
        // The scan stops past the PUSH data it skips, which for a PUSH truncated by the code end
        // is up to 32 positions past it. The map is not that long: clamp.
        m_scanned = std::min(
            scan_jumpdests(m_jumpdest_map, m_code.data(), m_scanned, position + 1), m_code.size());
    }
#endif
};

/// Analyze the EVM code in preparation for execution.
///
/// This builds the map of valid JUMPDESTs.
///
/// @param code         The reference to the EVM code to be analyzed.
EVMC_EXPORT CodeAnalysis analyze(bytes_view code);

/// Executes in Baseline interpreter using EVMC-compatible parameters.
evmc_result execute(evmc_vm* vm, const evmc_host_interface* host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message* msg, const uint8_t* code, size_t code_size) noexcept;

/// Executes in Baseline interpreter with the pre-processed code.
EVMC_EXPORT evmc_result execute(VM&, const evmc_host_interface& host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message& msg, const CodeAnalysis& analysis) noexcept;

}  // namespace baseline
}  // namespace evmone
