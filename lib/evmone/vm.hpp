// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2021 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "baseline.hpp"
#include "execution_state.hpp"
#include "tracing.hpp"
#include <evmc/evmc.h>

#include <cstddef>
#include <list>
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
    /// EVM call depth limit (1024) and max EVM stack items per frame (1024).
    /// The buffer covers MAX_DEPTH + 1 frame slots at depths 0..MAX_DEPTH: a frame at
    /// depth MAX_DEPTH still runs (it just cannot call deeper).
    static constexpr size_t MAX_DEPTH = 1024;
    static constexpr size_t MAX_STACK_PER_FRAME = 1024;

    /// Per-frame stride in bytes — exactly the live stack space.
    /// Frame N's bottom is at `stack_buffer_storage + N * FRAME_STRIDE`.
    static constexpr size_t FRAME_STRIDE = MAX_STACK_PER_FRAME * sizeof(intx::uint256);
    static_assert(FRAME_STRIDE == 0x8000);

    bool cgoto = EVMONE_CGOTO_SUPPORTED;

    /// The part of the buffer beyond the stack slots of all depths, lent to the frames'
    /// EVM memories.
    static constexpr size_t MEMORY_HEADROOM = 4 * 1024 * 1024;

    /// Shared EVM stack buffer. ~36 MiB ((MAX_DEPTH + 1) * 32 KB + MEMORY_HEADROOM).
    ///
    /// Frames are packed densely starting at `stack_buffer_storage`. Each
    /// frame's bottom is handed out as the current `stack_tail` — see
    /// `baseline::execute` for entry and `call_impl`/`create_impl` for the
    /// pre-call update. The spare space beyond a frame's stack is lent to
    /// that frame's EVM `Memory` as its initial borrowed backing, up to the
    /// stack slots kept free for the deeper frames, so that the frame at
    /// any depth up to MAX_DEPTH has its whole stack inside the buffer.
    ///
    /// Storage is a raw POD byte array so it is left uninitialized at VM
    /// construction (an `intx::uint256[]` member would otherwise zero-init
    /// every slot via its `words_[4]{}` member initializer).
    ///
    /// Single-threaded per VM.
    alignas(sizeof(intx::uint256)) std::byte
        stack_buffer_storage[(MAX_DEPTH + 1) * FRAME_STRIDE + MEMORY_HEADROOM];

    /// Next free byte in `stack_buffer_storage`. The next frame to be
    /// entered starts its stack at this address. Reset to the buffer base
    /// on depth-0 entry; updated by `call_impl`/`create_impl` before each
    /// nested call to skip past the parent's stack + live EVM memory.
    std::byte* stack_tail = stack_buffer_storage;

    /// One past the last byte of the shared buffer.
    [[nodiscard]] std::byte* stack_buffer_end() noexcept
    {
        return stack_buffer_storage + sizeof(stack_buffer_storage);
    }

private:
    std::vector<ExecutionState> m_execution_states;
    CodeCache m_code_cache;
    std::unique_ptr<Tracer> m_first_tracer;

public:
    VM() noexcept;

    // ExecutionState::stack_bottom points into this object's storage;
    // moving the VM would invalidate the pointer.
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
