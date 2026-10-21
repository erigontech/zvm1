// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2022 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "host.hpp"
#include "precompiles.hpp"
#include "system_contracts.hpp"
#include <evmone/constants.hpp>
#include <evmone/state_gas.hpp>
#include <evmone/vm.hpp>
#include <evmone/word_layout.hpp>
#include <cstddef>
#include <cstring>

namespace evmone::state
{
namespace
{
/// Whether the message runs the code of its recipient: msg.code_address == msg.recipient.
bool runs_recipient_code(const evmc_message& msg) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // Word by word, in place: operator== takes evmc::address, and converting the plain
    // evmc_address fields copies both. The fields are word aligned in the message.
    typedef uint32_t __attribute__((may_alias)) Word;
    static_assert(offsetof(evmc_message, recipient) % sizeof(Word) == 0 &&
                  offsetof(evmc_message, code_address) % sizeof(Word) == 0 &&
                  alignof(evmc_message) >= sizeof(Word) && sizeof(evmc_address) % sizeof(Word) == 0);
    const Word* const a = reinterpret_cast<const Word*>(msg.code_address.bytes);
    const Word* const b = reinterpret_cast<const Word*>(msg.recipient.bytes);
    for (size_t i = 0; i < sizeof(evmc_address) / sizeof(Word); ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
#else
    return std::memcmp(msg.code_address.bytes, msg.recipient.bytes, sizeof(evmc_address)) == 0;
#endif
}

/// The status and the journaled update of a slot's value, shared by set_storage() and sstore().
[[gnu::always_inline]] inline evmc_storage_status update_storage(
    State& state, StorageValue& storage_slot, const bytes32& value) noexcept
{
    // Follow EVMC documentation https://evmc.ethereum.org/storagestatus.html#autotoc_md3
    // and EIP-2200 specification https://eips.ethereum.org/EIPS/eip-2200.

    const auto& current = storage_slot.current;
    const auto& original = storage_slot.original;

    const auto dirty = original != current;
    const auto restored = original == value;
    const auto current_is_zero = is_zero(current);
    const auto value_is_zero = is_zero(value);

    auto status = EVMC_STORAGE_ASSIGNED;  // All other cases.
    if (!dirty && !restored)
    {
        if (current_is_zero)
            status = EVMC_STORAGE_ADDED;  // 0 → 0 → Z
        else if (value_is_zero)
            status = EVMC_STORAGE_DELETED;  // X → X → 0
        else
            status = EVMC_STORAGE_MODIFIED;  // X → X → Z
    }
    else if (dirty && !restored)
    {
        if (current_is_zero && !value_is_zero)
            status = EVMC_STORAGE_DELETED_ADDED;  // X → 0 → Z
        else if (!current_is_zero && value_is_zero)
            status = EVMC_STORAGE_MODIFIED_DELETED;  // X → Y → 0
    }
    else if (dirty)
    {
        assert(restored);  // Always true.
        if (current_is_zero)
            status = EVMC_STORAGE_DELETED_RESTORED;  // X → 0 → X
        else if (value_is_zero)
            status = EVMC_STORAGE_ADDED_DELETED;  // 0 → Y → 0
        else
            status = EVMC_STORAGE_MODIFIED_RESTORED;  // X → Y → X
    }

    state.journal_storage_change(storage_slot);
    storage_slot.current = value;  // Update current value.
    return status;
}
}  // namespace

bool Host::account_exists(const address& addr, void* acc_in) const noexcept
{
    const auto& acc = acc_in != nullptr ?
                          m_state.get(*static_cast<Account*>(acc_in), addr) :
                          m_state.get(addr);
    return !acc.nonexistent && (m_rev < EVMC_SPURIOUS_DRAGON || !acc.is_empty());
}

bytes32 Host::get_storage(const address& addr, const bytes32& key) const noexcept
{
    return m_state.get_storage(addr, key).current;
}

evmc_storage_status Host::set_storage(
    const address& addr, const bytes32& key, const bytes32& value) noexcept
{
    assert(!m_state.get(addr).nonexistent);
    return update_storage(m_state, m_state.get_storage(addr, key), value);
}

uint256be Host::get_balance(const address& addr, void* acc_in, void** acc_out) const noexcept
{
    auto& acc = acc_in != nullptr ? m_state.get(*static_cast<Account*>(acc_in), addr) :
                                    m_state.get(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;
    return intx::be::store<uint256be>(acc.balance);
}

uint64_t Host::get_nonce(const address& addr) const noexcept
{
    return m_state.get(addr).nonce;
}

namespace
{
/// Check if an existing account is the "create collision"
/// as defined in the [EIP-7610](https://eips.ethereum.org/EIPS/eip-7610).
[[nodiscard]] bool is_create_collision(const Account& acc) noexcept
{
    // TODO: This requires much more testing:
    // - what if an account had storage but is destructed?
    // - what if an account had cold storage but it was emptied?
    // - what if an account without cold storage gain one?
    if (acc.nonce != 0)
        return true;
    if (acc.code_hash != Account::EMPTY_CODE_HASH)
        return true;
    if (acc.has_initial_storage)
        return true;

    // The hot storage is ignored because it can contain elements from access list.
    // TODO: Is this correct for destructed accounts?
    //assert(!acc.destructed && "untested");
    return false;
}
}  // namespace

size_t Host::get_code_size(const address& addr) const noexcept
{
    const auto raw_code = m_state.get_code(addr);
    return raw_code.size();
}

bytes32 Host::get_code_hash(const address& addr) const noexcept
{
    const auto& acc = m_state.get(addr);
    if (acc.is_empty())
        return {};

    return acc.code_hash;
}

size_t Host::copy_code(const address& addr, size_t code_offset, uint8_t* buffer_data,
    size_t buffer_size, void* acc_in) const noexcept
{
    const auto code = acc_in != nullptr ?
                          m_state.get_code(*static_cast<Account*>(acc_in), addr) :
                          m_state.get_code(addr);
    const auto code_slice = code.substr(std::min(code_offset, code.size()));
    const auto num_bytes = std::min(buffer_size, code_slice.size());
    std::copy_n(code_slice.begin(), num_bytes, buffer_data);
    return num_bytes;
}

bool Host::selfdestruct(const address& addr, const address& beneficiary) noexcept
{
    auto& acc = m_state.get(addr);
    assert(!acc.nonexistent);
    const auto balance = acc.balance;
    auto& beneficiary_acc = m_state.touch(beneficiary);

    m_state.journal_balance_change(beneficiary_acc);
    m_state.journal_balance_change(acc);

    if (m_rev >= EVMC_CANCUN && !acc.just_created)
    {
        // EIP-6780:
        // "SELFDESTRUCT is executed in a transaction that is not the same
        // as the contract invoking SELFDESTRUCT was created"
        acc.balance = 0;
        beneficiary_acc.balance += balance;  // Keep balance if acc is the beneficiary.

        if (m_rev >= EVMC_AMSTERDAM)
            emit_transfer_log(m_logs, addr, beneficiary, balance);

        // Return "selfdestruct not registered".
        // In practice this affects only refunds before Cancun.
        return false;
    }

    if (m_rev < EVMC_AMSTERDAM || beneficiary != addr)
    {
        // Transfer may happen multiple times per single account as account's balance
        // can be increased with a call following previous selfdestruct.
        beneficiary_acc.balance += balance;
        acc.balance = 0;  // Zero balance if acc is the beneficiary (before EIP-8246)
    }

    if (m_rev >= EVMC_AMSTERDAM)
        emit_transfer_log(m_logs, addr, beneficiary, balance);

    // Mark the destruction if not done already.
    if (!acc.destructed)
    {
        m_state.journal_account_flags(acc);
        acc.destructed = true;
        return true;
    }
    return false;
}

evmc::Result Host::create(const evmc_message& msg) noexcept
{
    assert(msg.kind == EVMC_CREATE || msg.kind == EVMC_CREATE2);
    assert(msg.recipient != address{});  // Must be computed already.

    auto [new_acc, created] = m_state.get_or_create(msg.recipient);
    if (!created)
    {
        // TODO: Add EVMC errors for creation failures.
        if (is_create_collision(new_acc))
            return evmc::Result{EVMC_FAILURE, {.left = msg.state_gas}};
        m_state.journal_create(new_acc);
    }

    //assert(new_acc.nonce == 0);
    assert(!new_acc.just_created && !new_acc.code_changed);  // The create rollback clears them.

    if (m_rev >= EVMC_SPURIOUS_DRAGON)
        new_acc.nonce = 1;  // No need to journal: create revert will 0 the nonce.

    new_acc.just_created = true;

    auto& sender_acc = m_state.get(msg.sender);  // TODO: Duplicated account lookup.
    assert(!sender_acc.nonexistent);
    const auto value = intx::be::load<intx::uint256>(msg.value);
    //assert(sender_acc.balance >= value && "EVM must guarantee balance");
    m_state.journal_balance_change(sender_acc);
    m_state.journal_balance_change(new_acc);
    sender_acc.balance -= value;
    new_acc.balance += value;  // The new account may be prefunded.

    if (m_rev >= EVMC_AMSTERDAM)
        emit_transfer_log(m_logs, msg.sender, msg.recipient, value);

    auto create_msg = msg;
    create_msg.input_data = nullptr;
    create_msg.input_size = 0;
    const bytes_view initcode{msg.input_data, msg.input_size};
    m_frame_acc = &new_acc;  // Seed the initcode frame.
    auto result = m_vm.execute(*this, static_cast<evmc_revision>(m_rev), create_msg, initcode.data(), initcode.size());
    if (result.status_code != EVMC_SUCCESS)
        return result;

    auto gas_left = result.gas_left;
    //assert(gas_left >= 0);

    const bytes_view code{result.output_data, result.output_size};

    const size_t max_code_size = m_rev >= EVMC_AMSTERDAM ? MAX_CODE_SIZE_AMSTERDAM : MAX_CODE_SIZE;
    if (m_rev >= EVMC_SPURIOUS_DRAGON && code.size() > max_code_size)
        return evmc::Result{EVMC_FAILURE, {.left = msg.state_gas}};

    // Reject new contract code starting with the 0xEF byte (EIP-3541).
    if (m_rev >= EVMC_LONDON && code.starts_with(0xEF))
        return evmc::Result{EVMC_CONTRACT_VALIDATION_FAILURE, {.left = msg.state_gas}};

    StateGas state_gas{result.state_gas};  // The initcode's state-gas for code deposit.
    if (m_rev >= EVMC_AMSTERDAM)
    {
        // The code deposit splits into an execution-gas and a state-gas component (EIP-8037).
        // Unsigned: Airbender's v3 machine has no signed multiply, which a signed widening product
        // of the code size compiles to on rv32.
        const auto execution_cost = static_cast<int64_t>(6 * ((uint64_t{code.size()} + 31) / 32));
        const auto state_cost = static_cast<int64_t>(uint64_t{code.size()} * COST_PER_STATE_BYTE);
        gas_left -= execution_cost;
        if (gas_left < 0 || !state_gas.charge(gas_left, state_cost))
            return evmc::Result{EVMC_FAILURE, {.left = msg.state_gas}};
    }
    else
    {
        // Code deployment cost.
        const auto cost = std::ssize(code) * 200;
        gas_left -= cost;
        if (gas_left < 0)
        {
            return (m_rev == EVMC_FRONTIER) ?
                       evmc::Result{EVMC_SUCCESS, result.gas_left, result.gas_refund} :
                       evmc::Result{EVMC_FAILURE, {.left = msg.state_gas}};
        }
    }

    if (!code.empty())
    {
        new_acc.code_hash = keccak256(code);
        new_acc.code = code;
        new_acc.code_changed = true;
        m_state.add_changed_code_address(new_acc.code_hash, msg.recipient);
    }

    return evmc::Result{result.status_code, gas_left, result.gas_refund, state_gas};
}

evmc::Result Host::execute_message(const evmc_message& msg, void* acc_recipient,
    void* acc_code, void* acc_sender, bool value_checked, bool value_nonzero) noexcept
{
    if (msg.kind == EVMC_CREATE || msg.kind == EVMC_CREATE2)
        return create(msg);

    // The recipient's account of a call: both branches get it.
    Account* recipient_acc_ptr = nullptr;
    if (msg.kind == EVMC_CALL)
    {
        if (value_checked ? !value_nonzero : evmc::is_zero(msg.value))
        {
            recipient_acc_ptr = acc_recipient != nullptr ?
                                    &m_state.touch(*static_cast<Account*>(acc_recipient),
                                        msg.recipient) :
                                    &m_state.touch(msg.recipient);
        }
        else
        {
            // We skip touching if we send value, because account cannot end up empty.
            // It will either have value, or code that transfers this value out, or will be
            // selfdestructed anyway.
            auto& recipient_acc = acc_recipient != nullptr ?
                                      m_state
                                          .get_or_create(
                                              *static_cast<Account*>(acc_recipient), msg.recipient)
                                          .first :
                                      m_state.get_or_create(msg.recipient).first;
            recipient_acc_ptr = &recipient_acc;

            // Transfer value: sender → recipient.
            // The sender's balance is already checked therefore the sender account must exist.
            const auto value = intx::be::load<intx::uint256>(msg.value);
            auto& sender_acc = acc_sender != nullptr ?
                                   m_state.get(*static_cast<Account*>(acc_sender), msg.sender) :
                                   m_state.get(msg.sender);
            assert(!sender_acc.nonexistent);
            assert(sender_acc.balance >= value);
            m_state.journal_balance_change(sender_acc);
            m_state.journal_balance_change(recipient_acc);
            sender_acc.balance -= value;
            recipient_acc.balance += value;

            if (m_rev >= EVMC_AMSTERDAM)
                emit_transfer_log(m_logs, msg.sender, msg.recipient, value);
        }
    }

    // Calls to precompile address via EIP-7702 delegation execute empty code instead of precompile.
    if ((msg.flags & EVMC_DELEGATED) == 0 && is_precompile(static_cast<evmc_revision>(m_rev), msg.code_address))
        return call_precompile(static_cast<evmc_revision>(m_rev), msg);

    // A call that runs the recipient's own code (not a delegated one) has its account already,
    // and the node stays where it is through the inserts since.
    Account* code_acc;
    if (msg.kind == EVMC_CALL && runs_recipient_code(msg))
        code_acc = recipient_acc_ptr;
    else if (acc_code != nullptr)
        code_acc = &m_state.get(*static_cast<Account*>(acc_code), msg.code_address);
    else
        code_acc = &m_state.get(msg.code_address);
    // A nonexistent account has the empty code hash, so it needs no separate check.
    if (code_acc->code_hash == Account::EMPTY_CODE_HASH)  // Skip trivial execution.
        return evmc::Result{EVMC_SUCCESS, msg.gas, 0, {.left = msg.state_gas}};

    // Taken by the frame's reset().
    m_frame_acc = recipient_acc_ptr != nullptr ? recipient_acc_ptr : acc_recipient;
    auto* my_vm = static_cast<VM*>(m_vm.get_raw_pointer());
    if (my_vm->has_cached_execution())
    {
        return my_vm->execute_cached_code(*this, static_cast<evmc_revision>(m_rev), msg, code_acc->code_hash,
            [this](const address& addr) { return m_state.get_code(addr); });
    }

    const auto code = m_state.get_code(*code_acc, msg.code_address);
    if (code.empty())  // Skip trivial execution.
        return evmc::Result{EVMC_SUCCESS, msg.gas, 0, {.left = msg.state_gas}};

    return m_vm.execute(*this, static_cast<evmc_revision>(m_rev), msg, code.data(), code.size());
}

evmc::Result Host::call(const evmc_message& msg, void* acc_recipient,
    void* acc_code, void* acc_sender, bool value_checked, bool value_nonzero) noexcept
{
    if (msg.depth != 0 && (msg.kind == EVMC_CREATE || msg.kind == EVMC_CREATE2))
    {
        // Bump the creator's nonce (already done for depth 0). Not reverted if creation fails.
        auto& sender_acc = m_state.get(msg.sender);
        assert(!sender_acc.nonexistent);
        assert(sender_acc.nonce != MAX_NONCE);
        m_state.journal_bump_nonce(sender_acc);
        ++sender_acc.nonce;
    }

    const auto logs_checkpoint = m_logs.size();
    const auto state_checkpoint = m_state.checkpoint();

    auto result = execute_message(msg, acc_recipient, acc_code, acc_sender,
        value_checked, value_nonzero);

    if (result.status_code != EVMC_SUCCESS)
    {
        assert(result.state_gas.left == msg.state_gas);
        assert(result.state_gas.spilled == 0);

        // The 0x03 (RIPEMD-160) touch quirk: a touch on this address is
        // never reverted. It only matters when the account is empty, so gate it by rev range.
        static constexpr auto ADDR_03 = 0x03_address;
        const bool is_03_touched = m_rev < EVMC_PARIS && m_rev >= EVMC_SPURIOUS_DRAGON &&
                                   m_state.get(ADDR_03).erase_if_empty;

        // Revert.
        m_state.rollback(state_checkpoint);
        m_logs.resize(logs_checkpoint);

        if (is_03_touched) [[unlikely]]
            m_state.touch(ADDR_03);
    }
    return result;
}

evmc_tx_context Host::get_tx_context() const noexcept
{
    if (m_tx_context_set)
        return m_tx_context;
    // TODO: The effective gas price is already computed in transaction validation.
    // TODO: The effective gas price calculation is broken for system calls (gas price 0).
    assert(m_tx.max_gas_price >= m_block.base_fee || m_tx.max_gas_price == 0);
    const auto priority_gas_price =
        std::min(m_tx.max_priority_gas_price, m_tx.max_gas_price - m_block.base_fee);
    const auto effective_gas_price = m_block.base_fee + priority_gas_price;

    m_tx_context = evmc_tx_context{
        .tx_gas_price = intx::be::store<uint256be>(effective_gas_price),  // By EIP-1559.
        .tx_origin = m_tx.sender,
        .block_coinbase = m_block.coinbase,
        .block_number = m_block.number,
        .block_timestamp = m_block.timestamp,
        .block_gas_limit = m_block.gas_limit,
        .block_prev_randao = m_block.prev_randao,
        .chain_id = uint256be{m_block.chain_id},
        .block_base_fee = uint256be{m_block.base_fee},
        .blob_base_fee = intx::be::store<uint256be>(m_block.blob_base_fee.value_or(0)),
        .blob_hashes = m_tx.blob_hashes.data(),
        .blob_hashes_count = m_tx.blob_hashes.size(),
        .block_slot_number = m_block.slot_number.value_or(0),
    };
    m_tx_context_set = true;
    return m_tx_context;
}

bytes32 Host::get_block_hash(int64_t block_number) const noexcept
{
    return m_block_hashes.get_block_hash(block_number);
}

void Host::emit_log(const address& addr, const uint8_t* data, size_t data_size,
    const bytes32 topics[], size_t topics_count) noexcept
{
#ifdef EVMONE_WORD_LAYOUT
    // The data is a W pointer into the memory of the frame: one converting copy into the log.
    m_logs.push_back({addr, wl::to_bytes(data, data_size), {topics, topics + topics_count}});
#else
    m_logs.push_back({addr, {data, data_size}, {topics, topics + topics_count}});
#endif
}

evmc_access_status Host::access_account(const address& addr, void** acc_out) noexcept
{
    if (m_rev < EVMC_BERLIN)
        return EVMC_ACCESS_COLD;  // Ignore before Berlin.

    if (is_precompile(static_cast<evmc_revision>(m_rev), addr))  // Precompiles are always warm. Don't insert to state.
        return EVMC_ACCESS_WARM;

    auto& acc = m_state.access(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;  // Stable node: nothing erases accounts mid-transaction.
    if (acc.access_status == EVMC_ACCESS_WARM)
        return EVMC_ACCESS_WARM;

    m_state.journal_account_flags(acc);
    acc.access_status = EVMC_ACCESS_WARM;
    return EVMC_ACCESS_COLD;
}

evmc_access_status Host::access_storage(const address& addr, const bytes32& key) noexcept
{
    auto& storage_slot = m_state.access_storage(addr, key);
    if (storage_slot.access_status == EVMC_ACCESS_WARM)
        return EVMC_ACCESS_WARM;  // Nothing changes, skip journaling.
    m_state.journal_storage_change(storage_slot);
    storage_slot.access_status = EVMC_ACCESS_WARM;
    return EVMC_ACCESS_COLD;
}

const evmc_bytes32* Host::sload(const address& addr, const bytes32& key, int64_t cold_cost,
    int64_t& gas_left, evmc_bytes32& /*buffer*/, void* acc_in, void** acc_out) noexcept
{
    // access_storage() and get_storage() on one lookup of the account and of the slot (node
    // references are stable and nothing inserts between the two parts).
    // Recipient node; m_modified never erases mid-tx.
    auto& acc = acc_in != nullptr ? *static_cast<Account*>(acc_in) : m_state.access(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;
    assert(acc.loaded);  // The storage of the executing account, loaded to run its code.
    auto& slot = acc.storage[key];
    if (slot.access_status != EVMC_ACCESS_WARM)
    {
        m_state.journal_storage_change(slot);
        slot.access_status = EVMC_ACCESS_WARM;
        // Out of gas returns before the load: a witness may omit a slot that is never read.
        if ((gas_left -= cold_cost) < 0)
            return nullptr;
    }
    if (!slot.loaded)
        m_state.load_storage(addr, acc.view_handle, key, slot);
    return &slot.current;
}

evmc_storage_status Host::sstore(const address& addr, const bytes32& key, const bytes32& value,
    evmc_access_status& access, void* acc_in, void** acc_out) noexcept
{
    // access_storage() and set_storage() on one lookup of the account and of the slot, see sload().
    auto& acc = acc_in != nullptr ? *static_cast<Account*>(acc_in) : m_state.access(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;
    assert(acc.loaded && !acc.nonexistent);
    auto& slot = acc.storage[key];
    access = static_cast<evmc_access_status>(slot.access_status);
    if (access == EVMC_ACCESS_COLD)
    {
        m_state.journal_storage_change(slot);
        slot.access_status = EVMC_ACCESS_WARM;
    }
    if (!slot.loaded)
        m_state.load_storage(addr, acc.view_handle, key, slot);
    return update_storage(m_state, slot, value);
}


evmc::bytes32 Host::get_transient_storage(const address& addr, const bytes32& key,
    void* acc_in, void** acc_out) const noexcept
{
    auto& acc = acc_in != nullptr ? m_state.get(*static_cast<Account*>(acc_in), addr) :
                                    m_state.get(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;
    const auto it = acc.transient_storage.find(key);
    return it != acc.transient_storage.end() ? it->second : bytes32{};
}

void Host::set_transient_storage(const address& addr, const bytes32& key, const bytes32& value,
    void* acc_in, void** acc_out) noexcept
{
    auto& acc = acc_in != nullptr ? m_state.get(*static_cast<Account*>(acc_in), addr) :
                                    m_state.get(addr);
    if (acc_out != nullptr)
        *acc_out = &acc;
    assert(!acc.nonexistent);
    auto& slot = acc.transient_storage[key];
    m_state.journal_transient_storage_change(slot);
    slot = value;
}
}  // namespace evmone::state
