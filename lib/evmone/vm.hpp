// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2021 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "baseline.hpp"
#include "execution_state.hpp"
#include "tracing.hpp"
#include <evmc/evmc.h>

#include <list>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#define EVMONE_CGOTO_SUPPORTED 0
#else
#define EVMONE_CGOTO_SUPPORTED 1
#endif

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(EVMONE_RV32_DISPATCH_TEST)
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
    bool cgoto = EVMONE_CGOTO_SUPPORTED;

private:
    std::vector<ExecutionState> m_execution_states;
    CodeCache m_code_cache;
    std::unique_ptr<Tracer> m_first_tracer;

public:
    VM() noexcept;

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
