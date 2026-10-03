// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <evmc/utils.h>
#include <memory>

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(EVMONE_RV32_DISPATCH_TEST)
/// JUMPDEST analysis on demand: the scan runs up to the highest jump target checked so far
/// instead of over the whole code up front (10.7% of the analyzed bytes on mainnet lie past
/// every target). EVMONE_RV32_DISPATCH_TEST builds it on the host for testing.
#define EVMONE_LAZY_JUMPDESTS 1
#else
#define EVMONE_LAZY_JUMPDESTS 0
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
/// Classifies the code positions in [from, limit) (and the ones a PUSH's data carries the scan
/// past), setting the JUMPDEST bits in map, and returns the first position left unclassified.
/// The code must be the padded copy: the scan reads up to 7 bytes past limit. With limit at the
/// code size this is the whole analysis.
EVMC_EXPORT size_t scan_jumpdests(
    BitsetSpan map, const uint8_t* code, size_t from, size_t limit) noexcept;

class CodeAnalysis
{
private:
    bytes_view m_code;  ///< The executable code.

    /// Padded code for faster legacy code execution.
    /// If not nullptr m_code must point to it.
    std::unique_ptr<uint8_t[]> m_padded_code;

    BitsetSpan m_jumpdest_bitset{nullptr};
#if EVMONE_LAZY_JUMPDESTS
    mutable size_t m_scanned = 0;  ///< Positions below this are classified in the bitset.
#endif

public:
    /// Constructor for legacy code.
    CodeAnalysis(std::unique_ptr<uint8_t[]> padded_code, size_t code_size, BitsetSpan map)
      : m_code{padded_code.get(), code_size},
        m_padded_code{std::move(padded_code)},
        m_jumpdest_bitset{map}
    {}

    /// Constructor for legacy code whose padded copy starts inside the owned storage.
    CodeAnalysis(std::unique_ptr<uint8_t[]> storage, const uint8_t* padded_code,
        size_t code_size, BitsetSpan map)
      : m_code{padded_code, code_size}, m_padded_code{std::move(storage)}, m_jumpdest_bitset{map}
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
        if (position >= m_code.size())
            return false;
#if EVMONE_LAZY_JUMPDESTS
        if (position >= m_scanned) [[unlikely]]
            m_scanned = scan_jumpdests(
                m_jumpdest_bitset, m_code.data(), m_scanned, static_cast<size_t>(position) + 1);
#endif
        return m_jumpdest_bitset.test(static_cast<size_t>(position));
    }
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
