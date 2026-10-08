// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2025 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0
#include "delegation.hpp"
#include <cassert>

namespace evmone
{
namespace
{
/// Reads the delegation designation of addr, whose code starts with DELEGATION_MAGIC[0].
/// Kept out of line so that the one-byte probe in get_delegate_address(), which every CALL-like
/// instruction runs, stays small where it is inlined.
[[gnu::noinline, gnu::cold]] std::optional<evmc::address> read_designation(
    const evmc::HostInterface& host, const evmc::address& addr) noexcept
{
    // Load the code prefix up to the delegation designation size.
    // The HostInterface::copy_code() copies up to the addr's code size
    // and returns the number of bytes copied.
    uint8_t designation_buffer[std::size(DELEGATION_MAGIC) + sizeof(evmc::address)];
    const auto size = host.copy_code(addr, 0, designation_buffer, std::size(designation_buffer));
    const bytes_view designation{designation_buffer, size};

    if (!is_code_delegated(designation))
        return {};

    // Copy the delegate address from the designation buffer.
    evmc::address delegate_address;
    // Assume the designation with the valid magic has also valid length.
    // assert(designation.size() == std::size(designation_buffer));
    std::ranges::copy(designation.substr(std::size(DELEGATION_MAGIC)), delegate_address.bytes);
    return delegate_address;
}
}  // namespace

std::optional<evmc::address> get_delegate_address(
    const evmc::HostInterface& host, const evmc::address& addr) noexcept
{
    // Probe the first byte alone. Since EIP-3541, only delegation designations (and a few older
    // contracts) start with 0xEF, so almost every target is decided here. A one-byte copy is a
    // byte move in the host, while the full 23-byte prefix costs it a memmove call.
    uint8_t first_byte;
    if (host.copy_code(addr, 0, &first_byte, 1) == 0 || first_byte != DELEGATION_MAGIC[0])
        return {};

    return read_designation(host, addr);
}
}  // namespace evmone
