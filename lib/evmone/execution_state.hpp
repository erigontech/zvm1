// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2019 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "state_gas.hpp"
#include "word_layout.hpp"
#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace evmone
{
namespace advanced
{
struct AdvancedCodeAnalysis;
}
namespace baseline
{
class CodeAnalysis;
}

using evmc::bytes;
using evmc::bytes_view;
using intx::uint256;


/// Provides memory for EVM stack.
class StackSpace
{
    struct Storage
    {
        /// The maximum number of EVM stack items.
        static constexpr auto limit = 1024;

        /// Stack space items are aligned to 256 bits for better packing in cache lines.
        static constexpr auto alignment = sizeof(uint256);

        alignas(alignment) uint256 items[limit];
    };

    /// The storage allocated for maximum possible number of items.
    std::unique_ptr<Storage> m_stack_space = std::make_unique<Storage>();

public:
    static constexpr auto limit = Storage::limit;

    /// Returns the pointer to the "bottom", i.e. below the stack space.
    [[nodiscard]] uint256* bottom() noexcept { return &m_stack_space->items[0]; }
};


/// The EVM memory.
///
/// With EVMONE_WORD_LAYOUT the byte of EVM address a is at index a ^ 3 and the pointers into it
/// are W pointers, see word_layout.hpp. The allocation is 8-byte aligned, which the layout needs.
///
/// The implementations uses initial allocation of 4k and then grows capacity with 2x factor.
/// Some benchmarks have been done to confirm 4k is ok-ish value.
class Memory
{
    /// The size of allocation "page".
    static constexpr size_t page_size = 4 * 1024;

    struct FreeDeleter
    {
        void operator()(uint8_t* p) const noexcept { std::free(p); }
    };

    /// Owned pointer to allocated memory.
    std::unique_ptr<uint8_t[], FreeDeleter> m_data;

    /// The "virtual" size of the memory.
    size_t m_size = 0;

    /// The first offset a 32-byte access may not start at: m_size - 31, or 0 while the memory is
    /// empty (see limit32()). Written only together with m_size, by set_size().
    size_t m_lim = 0;

    /// The size of allocated memory. The initialization value is the initial capacity.
    size_t m_capacity = page_size;

    [[noreturn, gnu::cold]] static void handle_out_of_memory() noexcept { std::terminate(); }

    /// Sets the size, which is 0 or a multiple of 32, and its limit: m_size - 31, or 0 for the
    /// empty memory. Every change of m_size goes through here, and the caller names the limit so
    /// that it folds to a constant where the size is known (the one-word and the 96-byte growth).
    /// A limit that disagrees with the size is a read or write outside the memory (too large), or
    /// a growth to a size below the current one (too small).
    void set_size(size_t size, size_t limit) noexcept
    {
        assert(size % 32 == 0);
        assert(limit == (size != 0 ? size - 31 : 0));
        m_size = size;
        m_lim = limit;
    }

    /// Zeros the @p count 32-byte words at @p index, a multiple of 32 within the capacity.
    void zero_words(size_t index, size_t count) noexcept
    {
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
        // Word stores, as in grow(), rather than a memset call. One pointer for all of them: the
        // stores may alias m_data itself.
        wl::word_t* const w = reinterpret_cast<wl::word_t*>(&m_data[index]);
        for (size_t i = 0; i < 8 * count; ++i)
            w[i] = 0;
#else
        std::memset(&m_data[index], 0, 32 * count);
#endif
    }

    void allocate_capacity() noexcept
    {
        m_data.reset(static_cast<uint8_t*>(std::realloc(m_data.release(), m_capacity)));
        if (!m_data) [[unlikely]]
            handle_out_of_memory();
#ifdef EVMONE_WORD_LAYOUT
        assert(wl::is_aligned4(m_data.get()));
#endif
    }

public:
    /// Creates Memory object with initial capacity allocation.
    Memory() noexcept { allocate_capacity(); }

    uint8_t& operator[](size_t index) noexcept { return m_data[index]; }
    const uint8_t& operator[](size_t index) const noexcept { return m_data[index]; }

    [[nodiscard]] size_t size() const noexcept { return m_size; }

    /// The offset below which a 32-byte access lies inside the memory, whatever the offset: the
    /// size is a multiple of 32, so w + 32 <= size exactly when w < size - 31, which is one
    /// compare with no addition to wrap. The empty memory has the limit 0: every access to it
    /// grows it. An offset at or above the limit needs growth, or is out of gas.
    [[nodiscard]] size_t limit32() const noexcept
    {
        assert(m_lim == (m_size != 0 ? m_size - 31 : 0));
        return m_lim;
    }

    /// Grows the memory to the given size. The extent is filled with zeros.
    ///
    /// @param new_size  New memory size. Must be larger than the current size and multiple of 32.
    void grow(size_t new_size) noexcept
    {
        // Restriction for future changes. EVM always has memory size as multiple of 32 bytes.
        INTX_REQUIRE(new_size % 32 == 0);

        // Allow only growing memory. Include hint for optimizing compiler.
        INTX_REQUIRE(new_size > m_size);

        if (new_size > m_capacity)
        {
            m_capacity *= 2;  // Double the capacity.

            if (m_capacity < new_size)  // If not enough.
            {
                // Set capacity to required size rounded to multiple of page_size.
                m_capacity = ((new_size + (page_size - 1)) / page_size) * page_size;
            }

            allocate_capacity();
        }
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
        // Every size is a multiple of 32 and m_data comes from the 8-aligned allocator, so the new
        // extent is whole, word-aligned 32-byte words: store them inline, rather than calling
        // memset for the usual 1 to 3 of them.
        wl::word_t* w = reinterpret_cast<wl::word_t*>(&m_data[m_size]);
        wl::word_t* const end = reinterpret_cast<wl::word_t*>(&m_data[new_size]);
        // Not unrolled further: the growth is one 32-byte word 72% of the time, and GCC's 8-way
        // unrolling of this loop spent 8 instructions per call picking the remainder's entry.
#pragma GCC unroll 1
        do
        {
#pragma GCC unroll 8
            for (size_t i = 0; i < 8; ++i)
                w[i] = 0;
            w += 8;
        } while (w != end);
#else
        std::memset(&m_data[m_size], 0, new_size - m_size);
#endif
        set_size(new_size, new_size - 31);  // new_size exceeds the old size: it is at least 32.
    }

    /// The size of the allocation: never below its initial page.
    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }

    /// Grows the memory by one word for a 32-byte store at @p offset in (size() - 32, size()],
    /// within the capacity; the caller charges the gas. The store writes [offset, offset + 32)
    /// right after, so the new word needs zeros only when the store starts below the old end.
    void grow_word_for_store(size_t offset) noexcept
    {
        const auto old_size = m_size;  // Read once: the zeroing may alias it.
        assert(offset <= old_size && old_size < offset + 32 && old_size + 32 <= m_capacity);
        if (offset != old_size)
            zero_words(old_size, 1);
        set_size(old_size + 32, old_size + 1);
    }

    /// Grows the empty memory to 3 words for a 32-byte store at 0x40, which writes the third one
    /// right after: Solidity's free memory pointer initialization. The caller charges the gas.
    /// The buffer of a frame is reused at its depth, so the first two words are zeroed.
    void grow_empty_for_store_at_64() noexcept
    {
        static_assert(page_size >= 96, "the capacity is at least the initial page");
        assert(m_size == 0);
        zero_words(0, 2);
        set_size(96, 65);
    }

    /// Virtually clears the memory by setting its size to 0. The capacity stays unchanged.
    void clear() noexcept { set_size(0, 0); }
};

/// Generic execution state for generic instructions implementations.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class ExecutionState
{
public:
    int64_t gas_refund = 0;
    Memory memory;
    const evmc_message* msg = nullptr;
    /// The first calldata offset from which a 32-byte load is cut short: input_size - 31, or 0 for
    /// a calldata under 32 bytes. Set wherever msg is, so that CALLDATALOAD tests one compare for
    /// "the whole word is inside the input". 0 is the safe value: every load takes the full test.
    size_t cd_lim = 0;
    evmc::HostContext host;
    /// The C++ Host behind `host` when the frame runs through evmc::Host's own interface (null
    /// otherwise): SLOAD and SSTORE call its fused virtuals directly instead of several C callbacks,
    /// and CALL and CREATE receive its evmc::Result without the C round trip.
    evmc::Host* cpp_host = nullptr;
    evmc_revision rev = {};
#ifdef EVMONE_WORD_LAYOUT
    wl::ReturnData return_data;
    /// The init code of the frame's CREATE in byte order, for as long as the call that runs it.
    wl::Buffer init_code;
#else
    bytes return_data;
#endif

    /// Reference to original EVM code.
    bytes_view original_code;

    evmc_status_code status = EVMC_SUCCESS;
    size_t output_offset = 0;
    size_t output_size = 0;

private:
    evmc_tx_context m_tx = {};

public:
    /// Pointer to code analysis.
    /// This should be set and used internally by execute() function of a particular interpreter.
    union
    {
        const baseline::CodeAnalysis* baseline = nullptr;
        const advanced::AdvancedCodeAnalysis* advanced;
    } analysis{};

    /// The frame's state-gas counters (EIP-8037).
    StateGas state_gas;

    /// Stack space allocation.
    ///
    /// This is the last field to make other fields' offsets of reasonable values.
    StackSpace stack_space;

    ExecutionState() noexcept = default;

    ExecutionState(const evmc_message& message, evmc_revision revision,
        const evmc_host_interface& host_interface, evmc_host_context* host_ctx,
        bytes_view _code) noexcept
      : msg{&message},
        cd_lim{calldata_limit(message)},
        host{host_interface, host_ctx},
        cpp_host{cpp_host_of(host_interface, host_ctx)},
        rev{revision},
        original_code{_code},
        state_gas{{.left = message.state_gas}}
    {}

    /// Resets the contents of the ExecutionState so that it could be reused.
    void reset(const evmc_message& message, evmc_revision revision,
        const evmc_host_interface& host_interface, evmc_host_context* host_ctx,
        bytes_view _code) noexcept
    {
        gas_refund = 0;
        state_gas = {{.left = message.state_gas}};
        memory.clear();
        msg = &message;
        cd_lim = calldata_limit(message);
        host = {host_interface, host_ctx};
        cpp_host = cpp_host_of(host_interface, host_ctx);
        rev = revision;
        return_data.clear();
        original_code = _code;
        status = EVMC_SUCCESS;
        output_offset = 0;
        output_size = 0;
        // get_tx_context() refetches the whole context once block_timestamp is 0, and nothing
        // else reads m_tx: resetting that field invalidates it, where zeroing all 240 bytes was
        // a memset per message.
        m_tx.block_timestamp = 0;
    }

    [[nodiscard]] bool in_static_mode() const { return (msg->flags & EVMC_STATIC) != 0; }

    static size_t calldata_limit(const evmc_message& message) noexcept
    {
        return message.input_size >= 32 ? message.input_size - 31 : 0;
    }

    static evmc::Host* cpp_host_of(
        const evmc_host_interface& host_interface, evmc_host_context* host_ctx) noexcept
    {
        return &host_interface == &evmc::Host::get_interface() ?
                   evmc::Host::from_context(host_ctx) :
                   nullptr;
    }

    const evmc_tx_context& get_tx_context() noexcept
    {
        if (INTX_UNLIKELY(m_tx.block_timestamp == 0))
            m_tx = host.get_tx_context();
        return m_tx;
    }
};

/// Builds the execution result for a finished frame from its final @p state and @p gas_left.
///
/// Applies the frame-exit rules shared by the baseline and advanced interpreters: an exceptional
/// halt consumes all gas (only a success or revert keeps it), the gas refund counts only on
/// success, and the output is the memory range recorded in the state.
///
/// The result is returned as the evmc::Result the host hands to the calling frame: built in the
/// caller's return slot, it reaches call_impl with no release_raw() copy and no re-wrap. The C
/// entry points release it.
inline evmc::Result make_execution_result(ExecutionState& state, int64_t gas_left) noexcept
{
    if (state.rev >= EVMC_AMSTERDAM && state.status != EVMC_SUCCESS)
    {
        // Unsuccessful frame doesn't commit any state changes, roll-back all state-gas costs.
        gas_left += state.state_gas.spilled;
        state.state_gas.left = state.msg->state_gas;
        state.state_gas.spilled = 0;
    }

    // An exceptional halt consumes all gas; only a success or revert keeps gas_left.
    if (state.status != EVMC_SUCCESS && state.status != EVMC_REVERT)
        gas_left = 0;
    const auto gas_refund = (state.status == EVMC_SUCCESS) ? state.gas_refund : 0;

    assert(state.output_size != 0 || state.output_offset == 0);
#ifdef EVMONE_WORD_LAYOUT
    // One named result on every path, so that it is constructed in the return slot (NRVO).
    evmc::Result result{state.status, gas_left, gas_refund, state.state_gas};
    if (state.output_size != 0)
    {
        // The output leaves the frame in the layout its consumer reads: the W data of phase 0 for
        // the EVM frame that made the call, byte order for the host (a transaction, a system call
        // and the code a successful creation deploys). The storage of W data is whole words.
        const auto* const src = &state.memory[state.output_offset];
        const auto size = state.output_size;
        const bool deployed = state.status == EVMC_SUCCESS &&
                              (state.msg->kind == EVMC_CREATE || state.msg->kind == EVMC_CREATE2);
        const bool word_output = (state.msg->flags & wl::FLAG_WORD_OUTPUT) != 0 && !deployed;
        const auto storage = word_output ? wl::round_up4(size) : size;
        auto* const buffer = static_cast<uint8_t*>(std::malloc(storage));
        if (buffer == nullptr) [[unlikely]]
            std::terminate();
        if (word_output)
        {
            reinterpret_cast<wl::word_t*>(buffer)[storage / 4 - 1] = 0;
            wl::copy_w2w(buffer, src, size);
        }
        else
            wl::copy_w2b(buffer, src, size);
        auto& raw = result.raw();
        raw.output_data = buffer;
        raw.output_size = size;
        raw.release = evmc_free_result_memory;
    }
    return result;
#else
    return evmc::Result{state.status, gas_left, gas_refund,
        state.output_size != 0 ? &state.memory[state.output_offset] : nullptr, state.output_size,
        state.state_gas};
#endif
}
}  // namespace evmone
