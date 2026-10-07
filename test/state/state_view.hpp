// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2024 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <optional>

namespace evmone::state
{
using evmc::address;
using evmc::bytes;
using evmc::bytes32;
using evmc::bytes_view;
using intx::uint256;

class StateView
{
public:
    struct Account
    {
        uint64_t nonce = 0;
        uint256 balance;
        bytes32 code_hash;
        bool has_storage = false;

        /// Opaque to everyone but the StateView that set it: the view's own record of the
        /// account, passed back to get_storage_at() and get_account_code_at() so they need not
        /// look the account up again. Null when the view has no such record.
        const void* handle = nullptr;
    };

    virtual ~StateView() = default;
    virtual std::optional<Account> get_account(const address& addr) const noexcept = 0;
    /// The account code, borrowed. The returned view must stay valid while this StateView is
    /// alive and unmodified; an implementation that has no such storage must keep the bytes
    /// alive itself. Callers that need ownership copy at the point of use; most only inspect.
    virtual bytes_view get_account_code(const address& addr) const noexcept = 0;
    virtual bytes32 get_storage(const address& addr, const bytes32& key) const noexcept = 0;

    /// get_code() and get_storage() for an account whose get_account() returned @p handle (null
    /// if none). Named apart from the two above: same-name overloads make every override of
    /// only those hide these (-Woverloaded-virtual).
    virtual bytes_view get_account_code_at(
        const void* /*handle*/, const address& addr) const noexcept
    {
        return get_account_code(addr);
    }
    virtual bytes32 get_storage_at(
        const void* /*handle*/, const address& addr, const bytes32& key) const noexcept
    {
        return get_storage(addr, key);
    }
};


/// Interface to access hashes of known block headers.
class BlockHashes
{
public:
    virtual ~BlockHashes() = default;

    /// Returns the hash of the block header of the given block number.
    virtual bytes32 get_block_hash(int64_t block_number) const noexcept = 0;
};
}  // namespace evmone::state
