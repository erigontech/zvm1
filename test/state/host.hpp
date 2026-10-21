// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2022 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "state.hpp"
#include "state_view.hpp"
#include <evmone/create_address.hpp>

namespace evmone::state
{
using evmc::uint256be;

class Host : public evmc::Host
{
    evmc_revision m_rev;
    evmc::VM& m_vm;
    State& m_state;
    const BlockInfo& m_block;
    const BlockHashes& m_block_hashes;
    const Transaction& m_tx;
    std::vector<Log> m_logs;
    /// get_tx_context()'s answer, computed on the first call: the block and the transaction do not
    /// change while the Host runs them, and each frame of the transaction that reads the context
    /// (ORIGIN, GASPRICE, TIMESTAMP, ...) asks for it once. Not zeroed (250 bytes a transaction):
    /// it is read only once set.
    mutable evmc_tx_context m_tx_context;
    mutable bool m_tx_context_set = false;
    /// Next frame's recipient node, for reset().
    void* m_frame_acc = nullptr;

public:
    Host(evmc_revision rev, evmc::VM& vm, State& state, const BlockInfo& block,
        const BlockHashes& block_hashes, const Transaction& tx) noexcept
      : m_rev{rev}, m_vm{vm}, m_state{state}, m_block{block}, m_block_hashes{block_hashes}, m_tx{tx}
    {}

    [[nodiscard]] std::vector<Log>&& take_logs() noexcept { return std::move(m_logs); }

    void* take_frame_acc() noexcept override
    {
        const auto acc = m_frame_acc;
        m_frame_acc = nullptr;  // Stale nodes must never seed later frames.
        return acc;
    }

    evmc::Result call(const evmc_message& msg, void* acc_recipient = nullptr,
        void* acc_code = nullptr, void* acc_sender = nullptr) noexcept override;

    [[nodiscard]] bool account_exists(
        const address& addr, void* acc_in = nullptr) const noexcept override;

private:
    [[nodiscard]] bytes32 get_storage(
        const address& addr, const bytes32& key) const noexcept override;

    evmc_storage_status set_storage(
        const address& addr, const bytes32& key, const bytes32& value) noexcept override;

    [[nodiscard]] evmc::bytes32 get_transient_storage(const address& addr, const bytes32& key,
        void* acc_in = nullptr, void** acc_out = nullptr) const noexcept override;

    void set_transient_storage(const address& addr, const bytes32& key, const bytes32& value,
        void* acc_in = nullptr, void** acc_out = nullptr) noexcept override;

    [[nodiscard]] uint256be get_balance(const address& addr, void* acc_in = nullptr,
        void** acc_out = nullptr) const noexcept override;

    [[nodiscard]] uint64_t get_nonce(const address& addr) const noexcept override;

    [[nodiscard]] size_t get_code_size(const address& addr) const noexcept override;

    [[nodiscard]] bytes32 get_code_hash(const address& addr) const noexcept override;

    size_t copy_code(const address& addr, size_t code_offset, uint8_t* buffer_data,
        size_t buffer_size, void* acc_in = nullptr) const noexcept override;

    bool selfdestruct(const address& addr, const address& beneficiary) noexcept override;

    evmc::Result create(const evmc_message& msg) noexcept;

    [[nodiscard]] evmc_tx_context get_tx_context() const noexcept override;

    [[nodiscard]] bytes32 get_block_hash(int64_t block_number) const noexcept override;

    void emit_log(const address& addr, const uint8_t* data, size_t data_size,
        const bytes32 topics[], size_t topics_count) noexcept override;

public:
    evmc_access_status access_account(
        const address& addr, void** acc_out = nullptr) noexcept override;

private:
    evmc_access_status access_storage(const address& addr, const bytes32& key) noexcept override;

    const evmc_bytes32* sload(const address& addr, const bytes32& key, int64_t cold_cost,
        int64_t& gas_left, evmc_bytes32& buffer, void* acc_in, void** acc_out) noexcept override;

    evmc_storage_status sstore(const address& addr, const bytes32& key, const bytes32& value,
        evmc_access_status& access, void* acc_in = nullptr,
        void** acc_out = nullptr) noexcept override;

    evmc::Result execute_message(const evmc_message& msg, void* acc_recipient, void* acc_code,
        void* acc_sender) noexcept;
};
}  // namespace evmone::state
