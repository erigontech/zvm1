// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2019 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0

#include "constants.hpp"
#include "create_address.hpp"
#include "delegation.hpp"
#include "instructions.hpp"
#include <cstddef>

namespace evmone::instr::core
{
namespace
{
constexpr auto CALL_VALUE_COST = 9000;
constexpr auto CALL_VALUE_COST_AMSTERDAM = ACCOUNT_WRITE + CALL_STIPEND;
constexpr auto ACCOUNT_CREATION_COST = 25000;

/// The EIP-7702 delegation of the target of a code executing instruction.
enum class Delegation
{
    none,        ///< Not delegated, or delegated to itself: the target's own code runs.
    delegated,   ///< The delegate's code runs.
    out_of_gas,  ///< The access to the delegate account ran out of gas.
};

/// Resolves the EIP-7702 delegation of the target addr: if addr is delegated, writes the delegate
/// address to code_addr (which the caller has set to addr) and applies the gas charge for accessing
/// the delegate account, which may fail with out of gas.
inline Delegation resolve_delegation(const evmc::address& addr, evmc_address& code_addr,
    int64_t& gas_left, ExecutionState& state) noexcept
{
    if (state.rev < EVMC_PRAGUE)
        return Delegation::none;

    const auto delegate_addr = get_delegate_address(state.host, addr);
    if (!delegate_addr)
        return Delegation::none;

    const auto delegate_account_access_cost =
        (state.host.access_account(*delegate_addr) == EVMC_ACCESS_COLD ?
                cold_account_access(state.rev) :
                WARM_ACCESS);

    if ((gas_left -= delegate_account_access_cost) < 0)
        return Delegation::out_of_gas;

    // EIP-7928: once the access cost is committed (no OOG), the delegate
    // address must appear in the block access list even if the CALL itself
    // light-fails (e.g. insufficient funds) without doing any other state
    // touch on it. Force the lazy-load now so the BAL StateView decorator
    // observes the read.
    (void)state.host.account_exists(*delegate_addr);

    code_addr = *delegate_addr;
    return *delegate_addr != addr ? Delegation::delegated : Delegation::none;
}

/// Absorbs a child's state-gas back to the parent (EIP-8037).
inline void absorb_child_state_gas(
    int64_t& gas_left, ExecutionState& state, const evmc::Result& result) noexcept
{
    assert(result.state_gas.left >= 0);
    assert(result.state_gas.spilled >= 0);

    // At most one of the two pools is ever non-empty.
    assert(state.state_gas.left == 0 || state.state_gas.spilled == 0);
    assert(result.state_gas.left == 0 || result.state_gas.spilled == 0);

    // In a non-successful result, all is returned back.
    assert(result.status_code == EVMC_SUCCESS ||
           (result.state_gas.left == state.state_gas.left && result.state_gas.spilled == 0));

    // Accumulate the spilled state-gas.
    state.state_gas.spilled += result.state_gas.spilled;

    // Rebalance the state-gas refills: the caller must move callee's refills to gas_left up to the
    // caller's spilled counter. Do this by refilling all returned state-gas to zeroed `left`.
    state.state_gas.left = 0;
    state.state_gas.refill(gas_left, result.state_gas.left);
}

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || \
    defined(EVMONE_RV32_DISPATCH_TEST)
/// Whether the call family runs with the state Host only, as on rv32 (EVMONE_RV32_DISPATCH_TEST
/// builds that configuration on the host for testing), where no frame has state gas before
/// Amsterdam (see call_impl).
constexpr bool STATE_HOST_ONLY = true;
#else
constexpr bool STATE_HOST_ONLY = false;
#endif

/// The value operand of the instructions without one.
constexpr uint256 NO_VALUE{};

/// Writes the address in the low 20 bytes of x to dst, as intx::be::trunc<evmc::address>(x).
inline void to_address(evmc::address& dst, const uint256& x) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // Byte i of the big-endian address is byte 19 - i of x's little-endian words: one byte load
    // and store each, 40 instructions, where trunc byte-swaps all 32 bytes into a temporary and
    // then copies 20 of them. One asm block, as in intx's bswap256_bytes: GCC's bswap pass would
    // turn separate C byte copies back into word swaps of 10 instructions per word.
    using Bytes = uint8_t[32];
    const void* const src = &x;
    uint32_t t;
#define EVMONE_RB(si, di) "lbu %[t], " #si "(%[s])\n\tsb %[t], " #di "(%[d])\n\t"
    asm(EVMONE_RB(19, 0) EVMONE_RB(18, 1) EVMONE_RB(17, 2) EVMONE_RB(16, 3) EVMONE_RB(15, 4)
        EVMONE_RB(14, 5) EVMONE_RB(13, 6) EVMONE_RB(12, 7) EVMONE_RB(11, 8) EVMONE_RB(10, 9)
        EVMONE_RB(9, 10) EVMONE_RB(8, 11) EVMONE_RB(7, 12) EVMONE_RB(6, 13) EVMONE_RB(5, 14)
        EVMONE_RB(4, 15) EVMONE_RB(3, 16) EVMONE_RB(2, 17) EVMONE_RB(1, 18) EVMONE_RB(0, 19)
        : [t] "=&r"(t), "=m"(dst.bytes)
        : [d] "r"(dst.bytes), [s] "r"(src), "m"(*static_cast<const Bytes*>(src)));
#undef EVMONE_RB
#else
    dst = intx::be::trunc<evmc::address>(x);
#endif
}

/// Writes the big-endian value x to dst, a message's value.
inline void store_value(evmc_uint256be& dst, const uint256& x) noexcept
{
#if defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
    // The value of a message on the stack is 4-byte aligned, and a transferred amount has a few
    // significant words: the word-wise conversion stores each leading zero word as one zero word.
    static_assert(offsetof(evmc_message, value) % 4 == 0 && alignof(evmc_message) >= 4);
    intx::internal::bswap256_to_aligned(dst.bytes, &x);
#else
    intx::be::store(dst.bytes, x);
#endif
}
}  // namespace

/// Converts an opcode to matching EVMC call kind.
/// NOLINTNEXTLINE(misc-use-internal-linkage) fixed in clang-tidy 20.
consteval evmc_call_kind to_call_kind(Opcode op) noexcept
{
    switch (op)
    {
    case OP_CALL:
    case OP_STATICCALL:
        return EVMC_CALL;
    case OP_CALLCODE:
        return EVMC_CALLCODE;
    case OP_DELEGATECALL:
        return EVMC_DELEGATECALL;
    case OP_CREATE:
        return EVMC_CREATE;
    case OP_CREATE2:
        return EVMC_CREATE2;
    default:
        intx::unreachable();
    }
}

template <Opcode Op>
Result call_impl(StackTop stack, int64_t gas_left, ExecutionState& state) noexcept
{
    static_assert(
        Op == OP_CALL || Op == OP_CALLCODE || Op == OP_DELEGATECALL || Op == OP_STATICCALL);

    static constexpr bool HAS_VALUE_ARG = Op == OP_CALL || Op == OP_CALLCODE;

    // Built field by field, not value-initialized: that would clear all 144 bytes first (a memset
    // call on rv32), each to be written again. Every field is assigned before host.call() below;
    // only the padding is left, and nothing reads it.
    evmc_message msg;  // NOLINT(cppcoreguidelines-pro-type-member-init)

    // The operands are read in place, not copied: the stack slots stay valid. The push of the
    // result overwrites the last one (the output size), so it comes after the memory checks, which
    // read it last; the failures before it end the frame, whose stack and return data go with it.
    const auto& gas = stack.pop();
    evmc::address dst;
    to_address(dst, stack.pop());
    const auto& value = HAS_VALUE_ARG ? stack.pop() : NO_VALUE;
    const auto has_value = value != 0;
    const auto& input_offset_u256 = stack.pop();
    const auto& input_size_u256 = stack.pop();
    const auto& output_offset_u256 = stack.pop();
    const auto& output_size_u256 = stack.pop();

    if constexpr (Op == OP_CALL)
    {
        // TODO: gas_left is used as no-op and ignored by caller. Refactor this.
        if (has_value && state.in_static_mode())
            return {EVMC_STATIC_MODE_VIOLATION, gas_left};
    }

    if (!check_memory(gas_left, state.memory, input_offset_u256, input_size_u256))
        return {EVMC_OUT_OF_GAS, gas_left};

    if (!check_memory(gas_left, state.memory, output_offset_u256, output_size_u256))
        return {EVMC_OUT_OF_GAS, gas_left};

    const auto input_offset = static_cast<size_t>(input_offset_u256);
    const auto input_size = static_cast<size_t>(input_size_u256);
    const auto output_offset = static_cast<size_t>(output_offset_u256);
    const auto output_size = static_cast<size_t>(output_size_u256);

    stack.push(0);  // Assume failure.
    state.return_data.clear();

    if constexpr (HAS_VALUE_ARG)
    {
        const auto call_value_cost =
            state.rev >= EVMC_AMSTERDAM ? CALL_VALUE_COST_AMSTERDAM : CALL_VALUE_COST;
        if (has_value && (gas_left -= call_value_cost) < 0)
            return {EVMC_OUT_OF_GAS, gas_left};
    }

    if (state.rev >= EVMC_BERLIN && state.host.access_account(dst) == EVMC_ACCESS_COLD)
    {
        if ((gas_left -= additional_cold_account_access(state.rev)) < 0)
            return {EVMC_OUT_OF_GAS, gas_left};
    }

    msg.code_address = dst;
    const auto delegation = resolve_delegation(dst, msg.code_address, gas_left, state);
    if (delegation == Delegation::out_of_gas)
        return {EVMC_OUT_OF_GAS, gas_left};

    bool new_account_charged = false;  // NOLINT(*-const-correctness)
    if constexpr (Op == OP_CALL)
    {
        if ((has_value || state.rev < EVMC_SPURIOUS_DRAGON) && !state.host.account_exists(dst))
        {
            if (state.rev >= EVMC_AMSTERDAM)
            {
                if (!state.state_gas.charge(gas_left, NEW_ACCOUNT_STATE_GAS))
                    return {EVMC_OUT_OF_GAS, gas_left};
                new_account_charged = true;
            }
            else if ((gas_left -= ACCOUNT_CREATION_COST) < 0)
                return {EVMC_OUT_OF_GAS, gas_left};
        }
    }

    msg.kind = to_call_kind(Op);
    msg.flags = (Op == OP_STATICCALL) ?
                    uint32_t{EVMC_STATIC} :
                    state.msg->flags & ~std::underlying_type_t<evmc_flags>{EVMC_DELEGATED};
    if (delegation == Delegation::delegated)
        msg.flags |= EVMC_DELEGATED;
#ifdef EVMONE_WORD_LAYOUT
    // Set here, after the flags are known: STATICCALL replaced them and the others inherit them.
    msg.flags |= wl::FLAG_WORD_INPUT | wl::FLAG_WORD_OUTPUT;
#endif
    msg.depth = state.msg->depth + 1;
    msg.state_gas = state.state_gas.left;
    msg.recipient = (Op == OP_CALL || Op == OP_STATICCALL) ? dst : state.msg->recipient;
    msg.sender = (Op == OP_DELEGATECALL) ? state.msg->sender : state.msg->recipient;
    if constexpr (Op == OP_DELEGATECALL)
        msg.value = state.msg->value;
    else if (HAS_VALUE_ARG && has_value)
        store_value(msg.value, value);
    else
        msg.value = {};

    if (input_size > 0)
    {
        // input_offset may be garbage if input_size == 0.
        msg.input_data = &state.memory[input_offset];
        msg.input_size = input_size;
    }
    else
    {
        msg.input_data = nullptr;
        msg.input_size = 0;
    }
    msg.code = nullptr;
    msg.code_size = 0;

    msg.gas = std::numeric_limits<int64_t>::max();
    if (gas < msg.gas)
        msg.gas = static_cast<int64_t>(gas);

    if constexpr (Op == OP_STATICCALL)
    {
        msg.gas = std::min(msg.gas, gas_left - gas_left / 64);
    }
    else
    {
        if (state.rev >= EVMC_TANGERINE_WHISTLE)  // Always true for STATICCALL.
            msg.gas = std::min(msg.gas, gas_left - gas_left / 64);
        else if (msg.gas > gas_left)
            return {EVMC_OUT_OF_GAS, gas_left};
    }

    if constexpr (HAS_VALUE_ARG)
    {
        if (has_value)
        {
            msg.gas += CALL_STIPEND;
            gas_left += CALL_STIPEND;
            if (intx::be::load<uint256>(state.host.get_balance(state.msg->recipient)) < value)
            {
                if (new_account_charged)
                    state.state_gas.refill(gas_left, NEW_ACCOUNT_STATE_GAS);
                return {EVMC_SUCCESS, gas_left};  // "Light" failure.
            }
        }
    }

    if (state.rev < EVMC_OSAKA && state.msg->depth >= 1024)
        return {EVMC_SUCCESS, gas_left};  // "Light" failure.

    const auto result = state.host.call(msg);
    state.return_data.assign(result.output_data, result.output_size);
    stack.top() = result.status_code == EVMC_SUCCESS;

    if (const auto copy_size = std::min(output_size, result.output_size); copy_size > 0)
    {
#ifdef EVMONE_WORD_LAYOUT
        wl::copy_w2w(&state.memory[output_offset], result.output_data, copy_size);
#else
        std::memcpy(&state.memory[output_offset], result.output_data, copy_size);
#endif
    }

    const auto gas_used = msg.gas - result.gas_left;
    gas_left -= gas_used;
    state.gas_refund += result.gas_refund;
    // Before Amsterdam no frame has state gas: a transaction's message starts with none, and every
    // charge and refill is an Amsterdam rule, so the child returns none and absorbing it changes
    // nothing. Only the state Host is held to that; a test host may return state gas anyway.
    if (!STATE_HOST_ONLY || state.rev >= EVMC_AMSTERDAM)
        absorb_child_state_gas(gas_left, state, result);

    if constexpr (Op == OP_CALL)
    {
        if (new_account_charged && result.status_code != EVMC_SUCCESS)
            state.state_gas.refill(gas_left, NEW_ACCOUNT_STATE_GAS);
    }

    return {EVMC_SUCCESS, gas_left};
}

template Result call_impl<OP_CALL>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;
template Result call_impl<OP_STATICCALL>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;
template Result call_impl<OP_DELEGATECALL>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;
template Result call_impl<OP_CALLCODE>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;

template <Opcode Op>
Result create_impl(StackTop stack, int64_t gas_left, ExecutionState& state) noexcept
{
    static_assert(Op == OP_CREATE || Op == OP_CREATE2);

    if (state.in_static_mode())
        return {EVMC_STATIC_MODE_VIOLATION, gas_left};

    const auto endowment = stack.pop();
    const auto init_code_offset_u256 = stack.pop();
    const auto init_code_size_u256 = stack.pop();
    const auto salt = (Op == OP_CREATE2) ? intx::be::store<bytes32>(stack.pop()) : bytes32{};

    stack.push(0);  // Assume failure.
    state.return_data.clear();

    if (!check_memory(gas_left, state.memory, init_code_offset_u256, init_code_size_u256))
        return {EVMC_OUT_OF_GAS, gas_left};

    const auto init_code_offset = static_cast<size_t>(init_code_offset_u256);
    const auto init_code_size = static_cast<size_t>(init_code_size_u256);

    const size_t max_init_code_size =
        state.rev >= EVMC_AMSTERDAM ? MAX_INITCODE_SIZE_AMSTERDAM : MAX_INITCODE_SIZE;
    if (state.rev >= EVMC_SHANGHAI && init_code_size > max_init_code_size)
        return {EVMC_OUT_OF_GAS, gas_left};

    const auto init_code_word_cost = 6 * (Op == OP_CREATE2) + 2 * (state.rev >= EVMC_SHANGHAI);
    const auto init_code_cost = num_words(init_code_size) * init_code_word_cost;
    if ((gas_left -= init_code_cost) < 0)
        return {EVMC_OUT_OF_GAS, gas_left};

    if (state.rev < EVMC_OSAKA && state.msg->depth >= 1024)
        return {EVMC_SUCCESS, gas_left};  // "Light" failure.

    if (endowment != 0 &&
        intx::be::load<uint256>(state.host.get_balance(state.msg->recipient)) < endowment)
        return {EVMC_SUCCESS, gas_left};  // "Light" failure.

    const auto& sender = state.msg->recipient;
    const auto sender_nonce = state.host.get_nonce(sender);  // Pre-bump sender nonce.

    // Creation fails when the sender's nonce is at maximum (EIP-2681).
    if (sender_nonce == MAX_NONCE)
        return {EVMC_SUCCESS, gas_left};  // "Light" failure.

#ifdef EVMONE_WORD_LAYOUT
    // The init code is code: byte order, and it must outlive the call that runs it, in which the
    // frames below this one use their own buffers.
    const uint8_t* init_code_data = nullptr;
    if (init_code_size > 0)
    {
        auto* const buffer = state.init_code.get(init_code_size);
        wl::copy_w2b(buffer, &state.memory[init_code_offset], init_code_size);
        init_code_data = buffer;
    }
    const auto init_code = bytes_view{init_code_data, init_code_size};
#else
    const auto init_code =
        bytes_view{init_code_size > 0 ? &state.memory[init_code_offset] : nullptr, init_code_size};
#endif

    evmc_message msg{.kind = to_call_kind(Op)};
    msg.recipient = (Op == OP_CREATE) ? compute_create_address(sender, sender_nonce) :
                                        compute_create2_address(sender, salt, init_code);

    // Access to the new address is warmed and never reverted (EIP-2929).
    if (state.rev >= EVMC_BERLIN)
        state.host.access_account(msg.recipient);

    bool new_account_charged = false;
    if (state.rev >= EVMC_AMSTERDAM && !state.host.account_exists(msg.recipient))
    {
        if (!state.state_gas.charge(gas_left, NEW_ACCOUNT_STATE_GAS))
            return {EVMC_OUT_OF_GAS, gas_left};
        new_account_charged = true;
    }

    msg.gas = gas_left;
    if (state.rev >= EVMC_TANGERINE_WHISTLE)
        msg.gas -= msg.gas / 64;

    msg.state_gas = state.state_gas.left;
    msg.input_data = init_code.data();
    msg.input_size = init_code.size();
    msg.sender = sender;
    msg.depth = state.msg->depth + 1;
    msg.value = intx::be::store<evmc::uint256be>(endowment);
#ifdef EVMONE_WORD_LAYOUT
    msg.flags = wl::FLAG_WORD_OUTPUT;  // The REVERT data is read as return data.
#endif

    const auto result = state.host.call(msg);
    gas_left -= msg.gas - result.gas_left;
    state.gas_refund += result.gas_refund;
    absorb_child_state_gas(gas_left, state, result);
    if (new_account_charged && result.status_code != EVMC_SUCCESS)
        state.state_gas.refill(gas_left, NEW_ACCOUNT_STATE_GAS);

    state.return_data.assign(result.output_data, result.output_size);
    if (result.status_code == EVMC_SUCCESS)
        stack.top() = intx::be::load<uint256>(msg.recipient);

    return {EVMC_SUCCESS, gas_left};
}

template Result create_impl<OP_CREATE>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;
template Result create_impl<OP_CREATE2>(
    StackTop stack, int64_t gas_left, ExecutionState& state) noexcept;
}  // namespace evmone::instr::core
