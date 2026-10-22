// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2021 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "baseline.hpp"
#include "execution_state.hpp"
#include "tracing.hpp"
#include <evmc/evmc.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#define EVMONE_CGOTO_SUPPORTED 0
#else
#define EVMONE_CGOTO_SUPPORTED 1
#endif

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || \
    (defined(SP1) && defined(__riscv)) || defined(EVMONE_RV32_DISPATCH_TEST)
/// The code cache holds the analyses by value in the map, without LRU eviction: a guest runs one
/// block on a fresh VM, so the LRU eviction never fires on mainnet blocks and the LRU list, the
/// shared_ptr and the per-hit splice are pure overhead. EVMONE_RV32_DISPATCH_TEST builds it on
/// the host for testing; the host keeps the bounded LRU cache.
#define EVMONE_LEAN_CODE_CACHE 1
#else
#define EVMONE_LEAN_CODE_CACHE 0
#endif

namespace evmone
{
#if EVMONE_LEAN_CODE_CACHE
class CodeCache
{
    /// The map is node-based: the analysis of a running frame stays put while a nested call
    /// inserts (and rehashes). The key is the full 32-byte hash (the guest's std::hash only
    /// takes its last word, so equality must not rely on it).
    std::unordered_map<evmc::bytes32, baseline::CodeAnalysis> map_;

public:
    /// Returns the analysis of code_hash, made with analyze() when missing. The pointer stays
    /// valid for the lifetime of the cache.
    template <typename Analyze>
    const baseline::CodeAnalysis* get_or_analyze(const evmc::bytes32& code_hash, Analyze&& analyze)
    {
        const auto it = map_.find(code_hash);
        if (it != map_.end())
            return &it->second;
        return &map_.emplace(code_hash, analyze()).first->second;
    }
};
#else
class CodeCache
{
    // TODO: Make configurable by VM API.
    static constexpr size_t SIZE = 5000;

    using LRUList = std::list<std::pair<evmc::bytes32, std::shared_ptr<baseline::CodeAnalysis>>>;
    LRUList lru_list_;
    std::unordered_map<evmc::bytes32, LRUList::iterator> map_;

public:
    std::shared_ptr<baseline::CodeAnalysis> get(const evmc::bytes32& code_hash);

    void put(const evmc::bytes32& code_hash, std::shared_ptr<baseline::CodeAnalysis> code);

    template <typename Analyze>
    std::shared_ptr<baseline::CodeAnalysis> get_or_analyze(
        const evmc::bytes32& code_hash, Analyze&& analyze)
    {
        auto p = get(code_hash);
        if (p == nullptr)
        {
            p = std::make_shared<baseline::CodeAnalysis>(analyze());
            put(code_hash, p);
        }
        return p;
    }
};
#endif

/// The evmone EVMC instance.
class VM : public evmc_vm
{
public:
    /// Max EVM stack items per frame (1024).
    static constexpr size_t MAX_STACK_PER_FRAME = 1024;

    /// The stack space of a frame in bytes: its EVM memory starts after it.
    static constexpr size_t FRAME_STRIDE = MAX_STACK_PER_FRAME * sizeof(intx::uint256);
    static_assert(FRAME_STRIDE == 0x8000);

    bool cgoto = EVMONE_CGOTO_SUPPORTED;

    /// The most EVM memory a frame reserves in a stack segment, 4 MiB: the memory that the gas
    /// limit of a transaction (2^24, EIP-7825) can pay for is below it. A frame with more gas
    /// whose memory grows past the reservation moves it to the heap.
    static constexpr size_t MAX_RESERVED_MEMORY = size_t{4} * 1024 * 1024;

    /// The size of a stack segment: a frame with the largest reservation fits after almost
    /// 4 MiB of the frames calling it.
    static constexpr size_t STACK_SEGMENT_SIZE = size_t{8} * 1024 * 1024;
    static_assert(FRAME_STRIDE + MAX_RESERVED_MEMORY <= STACK_SEGMENT_SIZE);

    /// The upper bound of the EVM memory size in bytes that a frame with the gas limit @p gas can
    /// pay for, capped at MAX_RESERVED_MEMORY. The w words cost 3w + w²/512 gas, more than
    /// w²/512, so w < 16 * sqrt(2 * gas). The sqrt is bounded from above by one Newton step from
    /// a power of 2, with no multiplication nor division: off by at most 25%.
    [[nodiscard]] static constexpr size_t max_memory_size(int64_t gas) noexcept
    {
        // 16 * sqrt(2 * 2^25) words are MAX_RESERVED_MEMORY.
        if (gas >= (int64_t{1} << 25))
            return MAX_RESERVED_MEMORY;
        if (gas <= 0)
            return 0;
        const auto n = static_cast<uint32_t>(gas) * 2;
        const auto k = static_cast<unsigned>(std::bit_width(n) + 1) / 2;  // 2^k >= sqrt(n).
        // From above, the Newton step stays above sqrt(n) - 1.
        const auto r = (((uint32_t{1} << k) + (n >> k)) >> 1) + 1;
        return std::min(size_t{r} * 16 * 32, MAX_RESERVED_MEMORY);
    }

    /// The EVM stacks and memories of the frames, packed densely in stack segments. A frame
    /// starts at the stack tail: its stack takes FRAME_STRIDE and its EVM memory the rest of the
    /// segment, of which max_memory_size(msg.gas) is reserved: the frame moves to the next
    /// segment if the reservation does not fit. A call places the sub-frame right after the live
    /// EVM memory of the caller, which does not grow until the sub-frame returns. See
    /// `baseline::execute` for the entry and `call_impl`/`create_impl` for the pre-call update.
    ///
    /// The segments are kept after the frames in them return: a call in a loop at the end of a
    /// segment would otherwise allocate and free the next one on every iteration. They are
    /// malloc-allocated, so their pages are not touched before a frame uses them (a
    /// `intx::uint256[]` would zero-init every slot via its `words_[4]{}` member initializer).
    ///
    /// Single-threaded per VM.
    struct StackSegment
    {
        struct FreeDeleter
        {
            void operator()(std::byte* p) const noexcept { std::free(p); }
        };

        using Storage = std::unique_ptr<std::byte, FreeDeleter>;

        Storage storage;
        std::byte* begin = nullptr;  ///< The storage aligned to sizeof(uint256).
        std::byte* end = nullptr;
    };

    std::vector<StackSegment> stack_segments;

    /// The index of the segment of the stack tail.
    size_t stack_segment = 0;

    /// Next free byte in the current stack segment. The next frame to be entered starts its
    /// stack at this address, unless its reservation does not fit. Reset to the first segment
    /// on depth-0 entry; updated by `call_impl`/`create_impl` before each nested call to skip
    /// past the parent's stack + live EVM memory.
    std::byte* stack_tail = nullptr;

    /// Hands the stack tail after the caller's stack and live EVM memory to the sub-frame.
    void set_stack_tail(const ExecutionState& caller) noexcept
    {
        // With the memory moved to the heap, the segment space after the stack is free again.
        stack_segment = caller.stack_segment;
        stack_tail = caller.memory.is_borrowed() ?
                         reinterpret_cast<std::byte*>(caller.memory.borrowed_end()) :
                         reinterpret_cast<std::byte*>(caller.stack_bottom) + FRAME_STRIDE;
    }

    /// Places a frame reserving @p size bytes: at the stack tail if it fits in the current
    /// segment, otherwise at the start of the next segment (allocated if needed).
    /// Returns the frame bottom; the frame's space ends at the current segment's end.
    [[nodiscard]] std::byte* place_frame(size_t size) noexcept
    {
        if (size > static_cast<size_t>(stack_segments[stack_segment].end - stack_tail))
            [[unlikely]]
            next_stack_segment();
        return stack_tail;
    }

private:
    /// Moves the stack tail to the start of the next stack segment, allocating it if needed.
    [[gnu::cold]] void next_stack_segment() noexcept;

    std::vector<ExecutionState> m_execution_states;
    CodeCache m_code_cache;
    std::unique_ptr<Tracer> m_first_tracer;

public:
    VM() noexcept;

    // ExecutionState::vm points to this object; moving the VM would invalidate the pointer.
    VM(const VM&) = delete;
    VM(VM&&) = delete;
    VM& operator=(const VM&) = delete;
    VM& operator=(VM&&) = delete;

    /// Whether execute_cached_code() applies: only the Baseline interpreter caches analyses.
    [[nodiscard]] bool has_cached_execution() const noexcept;

    /// Executes msg in the Baseline interpreter with the cached analysis of code_hash (made with
    /// get_code when missing). The result is built in the caller's return slot: a raw
    /// evmc_result cost a release_raw() copy and a re-wrap per message, and a
    /// std::optional<evmc::Result> two moves. get_code is a template parameter, not a
    /// std::function: the call is resolved in the caller's lambda with no type erasure.
    template <typename GetCode>
    evmc::Result execute_cached_code(evmc::Host& host, evmc_revision rev, const evmc_message& msg,
        const evmc::bytes32& code_hash, GetCode&& get_code) noexcept
    {
        const auto p = m_code_cache.get_or_analyze(
            code_hash, [&] { return baseline::analyze(get_code(msg.code_address)); });
        return baseline::execute(
            *this, evmc::Host::get_interface(), host.to_context(), rev, msg, *p);
    }

    [[nodiscard]] ExecutionState& get_execution_state(size_t depth) noexcept;

    void add_tracer(std::unique_ptr<Tracer> tracer) noexcept
    {
        // Find the first empty unique_ptr and assign the new tracer to it.
        auto* end = &m_first_tracer;
        while (*end)
            end = &(*end)->m_next_tracer;
        *end = std::move(tracer);
    }

    void remove_tracers() noexcept { m_first_tracer.reset(); }

    [[nodiscard]] Tracer* get_tracer() const noexcept { return m_first_tracer.get(); }
};
}  // namespace evmone
