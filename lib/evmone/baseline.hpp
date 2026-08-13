// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <evmc/utils.h>
#include <memory>

namespace evmone
{
using evmc::bytes_view;
class ExecutionState;
class VM;

/// A span type for a bitset.
struct BitsetSpan
{
    using word_type = uint64_t;
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
class CodeAnalysis
{
private:
    bytes_view m_code;  ///< The executable code.

    /// Padded code for faster legacy code execution.
    /// If not nullptr m_code must point to it.
    /// Owns whatever this analysis allocated: the padded code copy for the owning
    /// constructor, or just the jumpdest bitset when the code is borrowed.
    std::unique_ptr<uint8_t[]> m_storage;

    BitsetSpan m_jumpdest_bitset{nullptr};

public:
    /// Constructor for legacy code left in the caller's buffer, which must outlive the
    /// analysis and carry the padding analyze_legacy() would have added.
    CodeAnalysis(bytes_view code, std::unique_ptr<uint8_t[]> storage, BitsetSpan map)
      : m_code{code}, m_storage{std::move(storage)}, m_jumpdest_bitset{map}
    {}

    /// Constructor for legacy code.
    CodeAnalysis(std::unique_ptr<uint8_t[]> padded_code, size_t code_size, BitsetSpan map)
      : m_code{padded_code.get(), code_size},
        m_storage{std::move(padded_code)},
        m_jumpdest_bitset{map}
    {}

    /// The executable code. This is where the interpreter should start execution.
    [[nodiscard]] bytes_view code() const noexcept { return m_code; }

    /// Check if given position is valid jump destination. Use only for legacy code.
    [[nodiscard]] bool check_jumpdest(uint64_t position) const noexcept
    {
        if (position >= m_code.size())
            return false;
        return m_jumpdest_bitset.test(static_cast<size_t>(position));
    }
};

/// The code padding the interpreter relies on: 32 bytes for the data of a PUSH32 truncated by
/// the code end, plus one more for a STOP terminating the code. Buffers executed in place must
/// carry this many trailing zero bytes; analyze() adds it to its own copy.
constexpr size_t CODE_PADDING = 32 + 1;

/// Analyze the EVM code in preparation for execution.
///
/// This builds the map of valid JUMPDESTs.
///
/// @param code         The reference to the EVM code to be analyzed.
EVMC_EXPORT CodeAnalysis analyze(bytes_view code);

/// analyze() that does not copy the code. See the borrowing CodeAnalysis constructor.
EVMC_EXPORT CodeAnalysis analyze_no_copy(bytes_view code);

/// Executes in Baseline interpreter using EVMC-compatible parameters.
evmc_result execute(evmc_vm* vm, const evmc_host_interface* host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message* msg, const uint8_t* code, size_t code_size) noexcept;

/// Executes in Baseline interpreter with the pre-processed code.
EVMC_EXPORT evmc_result execute(VM&, const evmc_host_interface& host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message& msg, const CodeAnalysis& analysis) noexcept;

}  // namespace baseline
}  // namespace evmone
