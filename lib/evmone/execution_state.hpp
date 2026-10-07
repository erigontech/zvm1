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

    /// The size of allocated memory. The initialization value is the initial capacity.
    size_t m_capacity = page_size;

    [[noreturn, gnu::cold]] static void handle_out_of_memory() noexcept { std::terminate(); }

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
        m_size = new_size;
    }

    /// Virtually clears the memory by setting its size to 0. The capacity stays unchanged.
    void clear() noexcept { m_size = 0; }
};

/// Generic execution state for generic instructions implementations.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class ExecutionState
{
public:
    int64_t gas_refund = 0;
    Memory memory;
    const evmc_message* msg = nullptr;
    evmc::HostContext host;
    /// The C++ Host behind `host` when the frame runs through evmc::Host's own interface (null
    /// otherwise): SLOAD and SSTORE call its fused virtuals directly instead of several C callbacks.
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
inline evmc_result make_execution_result(ExecutionState& state, int64_t gas_left) noexcept
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
        evmc_result result{};
        result.status_code = state.status;
        result.gas_left = gas_left;
        result.gas_refund = gas_refund;
        result.output_data = buffer;
        result.output_size = size;
        result.release = evmc_free_result_memory;
        result.state_gas = state.state_gas;
        return result;
    }
#endif
    return evmc::Result{state.status, gas_left, gas_refund,
        state.output_size != 0 ? &state.memory[state.output_offset] : nullptr, state.output_size,
        state.state_gas}
        .release_raw();
}
}  // namespace evmone
