// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2020 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "baseline.hpp"
#include "baseline_instruction_table.hpp"
#include "execution_state.hpp"
#include "instructions.hpp"
#include "zilk_core/print.hpp"
#include "vm.hpp"

#include <memory>

#ifdef NDEBUG
#define release_inline gnu::always_inline, msvc::forceinline
#else
#define release_inline
#endif

#ifdef __GNUC__
#define ASM_COMMENT(COMMENT) asm("# " #COMMENT)  // NOLINT(hicpp-no-assembler)
#else
#define ASM_COMMENT(COMMENT)
#endif

#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || defined(EVMONE_RV32_DISPATCH_TEST)
/// The rv32 dispatch paths: the 32-bit gas deduction, PUSH2+JUMP/JUMPI fusion and the folded
/// landing JUMPDEST. EVMONE_RV32_DISPATCH_TEST builds them on the host for testing.
#define EVMONE_RV32_DISPATCH 1
#else
#define EVMONE_RV32_DISPATCH 0
#endif

namespace evmone::baseline
{
namespace
{
/// Checks instruction requirements before execution.
///
/// This checks:
/// - if the instruction is defined
/// - if stack height requirements are fulfilled (stack overflow, stack underflow)
/// - charges the instruction base gas cost and checks is there is any gas left.
///
/// @tparam         Op            Instruction opcode.
/// @param          cost_table    Table of base gas costs.
/// The stack overflow limit, StackSpace::limit items above stack_bottom. On rv32 the 32 KiB
/// offset takes lui + add, which GCC rematerializes at every overflow check instead of keeping the
/// pointer in a register; hiding its value behind an empty asm makes it compute the limit once.
[[gnu::always_inline]] inline const uint256* stack_limit_of(const uint256* stack_bottom) noexcept
{
    const uint256* limit = stack_bottom + StackSpace::limit;
#if EVMONE_RV32_DISPATCH
    asm("" : "+r"(limit));
#endif
    return limit;
}

#if EVMONE_RV32_DISPATCH
/// gas_left -= cost; false once that is negative. cost is a non-negative 16-bit value.
///
/// On rv32 the int64 subtract-and-test is 6 instructions. Subtract from the low word and test
/// its sign: 2 instructions. A borrow always leaves the low word negative (at least
/// 2^32 - 2^15). A non-negative low word therefore borrowed nothing and is the exact 64-bit
/// result, still non-negative. A negative one (a borrow, or a low word of 2^31 or more) takes
/// the full 64-bit path, which recovers the borrow from the new low word alone.
[[gnu::always_inline]] inline bool deduct_gas(int64_t& gas_left, uint32_t cost) noexcept
{
    const auto g = static_cast<uint64_t>(gas_left);
    auto lo = static_cast<uint32_t>(g) - cost;
    asm("" : "+r"(lo));  // Keep GCC from folding lo + cost below back into the old low word.
    if (static_cast<int32_t>(lo) >= 0) [[likely]]
    {
        gas_left = static_cast<int64_t>((g & 0xffffffff00000000) | lo);
        return true;
    }
    const auto borrow = static_cast<uint32_t>(lo + cost < cost);
    const auto hi = static_cast<uint32_t>(g >> 32) - borrow;
    gas_left = static_cast<int64_t>((uint64_t{hi} << 32) | lo);
    return gas_left >= 0;
}
#endif

/// @param [in,out] gas_left      Gas left.
/// @param          stack_top     Pointer to the stack top item.
/// @param          stack_bottom  Pointer to the stack bottom.
///                               The stack height is stack_top - stack_bottom.
/// @return  Status code with information which check has failed
///          or EVMC_SUCCESS if everything is fine.
template <Opcode Op>
inline evmc_status_code check_requirements(const CostTable& cost_table, int64_t& gas_left,
    const uint256* stack_top, const uint256* stack_bottom, const uint256* stack_limit) noexcept
{
    static_assert(
        !instr::has_const_gas_cost(Op) || instr::gas_costs[EVMC_FRONTIER][Op] != instr::undefined,
        "undefined instructions must not be handled by check_requirements()");

    static_assert(
        !instr::has_const_gas_cost(Op) || instr::gas_costs[EVMC_FRONTIER][Op] != instr::undefined,
        "undefined instructions must not be handled by check_requirements()");

    auto gas_cost = instr::gas_costs[EVMC_FRONTIER][Op];  // Init assuming const cost.
    if constexpr (!instr::has_const_gas_cost(Op))
    {
        gas_cost = cost_table[Op];  // If not, load the cost from the current revision cost table.

        // Negative cost marks an undefined instruction.
        // This check must be the first to produce the correct error code.
        // By definition not possible if defined since the first revision.
        if constexpr (instr::traits[Op].since != EVMC_FRONTIER)
        {
            if (INTX_UNLIKELY(gas_cost < 0))
                return EVMC_UNDEFINED_INSTRUCTION;
        }
    }

    // Check stack requirements first. This order is not required,
    // but it is nicer because a complete gas check may need to inspect operands.
    if constexpr (instr::traits[Op].stack_height_change > 0)
    {
        static_assert(instr::traits[Op].stack_height_change == 1,
            "unexpected instruction with multiple results");
        if (INTX_UNLIKELY(stack_top == stack_limit))
            return EVMC_STACK_OVERFLOW;
    }
    if constexpr (instr::traits[Op].stack_height_required > 0)
    {
        // Check stack underflow using pointer comparison <= (better optimization).
        static constexpr auto min_offset = instr::traits[Op].stack_height_required - 1;
        if (INTX_UNLIKELY(stack_top <= stack_bottom + min_offset))
            return EVMC_STACK_UNDERFLOW;
    }

    if constexpr (!instr::has_const_gas_cost(Op) || instr::gas_costs[EVMC_FRONTIER][Op] > 0)
    {
#if EVMONE_RV32_DISPATCH
        if (INTX_UNLIKELY(!deduct_gas(gas_left, static_cast<uint32_t>(gas_cost))))
            return EVMC_OUT_OF_GAS;
#else
        if (INTX_UNLIKELY((gas_left -= gas_cost) < 0))
            return EVMC_OUT_OF_GAS;
#endif
    }

    return EVMC_SUCCESS;
}


/// The execution position.
struct Position
{
    code_iterator code_it;  ///< The position in the code.
    uint256* stack_end;     ///< The pointer to the stack end.
};

/// Helpers for invoking instruction implementations of different signatures.
/// @{
[[release_inline]] inline code_iterator invoke(void (*instr_fn)(StackTop) noexcept, Position pos,
    int64_t& /*gas*/, ExecutionState& /*state*/) noexcept
{
    instr_fn(pos.stack_end);
    return pos.code_it + 1;
}

[[release_inline]] inline code_iterator invoke(
    Result (*instr_fn)(StackTop, int64_t, ExecutionState&) noexcept, Position pos, int64_t& gas,
    ExecutionState& state) noexcept
{
    const auto o = instr_fn(pos.stack_end, gas, state);
    gas = o.gas_left;
    if (o.status != EVMC_SUCCESS)
    {
        state.status = o.status;
        return nullptr;
    }
    return pos.code_it + 1;
}

[[release_inline]] inline code_iterator invoke(void (*instr_fn)(StackTop, ExecutionState&) noexcept,
    Position pos, int64_t& /*gas*/, ExecutionState& state) noexcept
{
    instr_fn(pos.stack_end, state);
    return pos.code_it + 1;
}

[[release_inline]] inline code_iterator invoke(
    code_iterator (*instr_fn)(StackTop, ExecutionState&, code_iterator) noexcept, Position pos,
    int64_t& /*gas*/, ExecutionState& state) noexcept
{
    return instr_fn(pos.stack_end, state, pos.code_it);
}

[[release_inline]] inline code_iterator invoke(
    TermResult (*instr_fn)(StackTop, int64_t, ExecutionState&) noexcept, Position pos, int64_t& gas,
    ExecutionState& state) noexcept
{
    const auto result = instr_fn(pos.stack_end, gas, state);
    gas = result.gas_left;
    state.status = result.status;
    return nullptr;
}

#if EVMONE_RV32_DISPATCH
/// Charges and steps over the JUMPDEST a jump lands on: it does nothing but cost 1 gas, and the
/// destination is a JUMPDEST by construction. Running it separately would be another dispatch
/// (6 instructions) and gas check. Only the status of a failure is observable, so charging the
/// 1 gas here changes nothing: out of gas is out of gas, at the JUMPDEST or one step earlier.
[[gnu::always_inline]] inline code_iterator skip_landing_jumpdest(
    code_iterator target, int64_t& gas, ExecutionState& state) noexcept
{
    if (INTX_UNLIKELY(!deduct_gas(gas, 1)))
    {
        state.status = EVMC_OUT_OF_GAS;
        return nullptr;
    }
    return target + 1;
}

/// PUSH2 followed by JUMP or JUMPI, which is how nearly every jump is written (99.6% of them on
/// mainnet). The destination is the immediate, so it is never stored to the stack and read back
/// and its high words need no zero check, and the landing JUMPDEST is folded in. The checks run
/// in the order the separate instructions would run them, so a failure gets the same status.
[[gnu::always_inline]] inline bool fused_push2_jump(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto op = pos.code_it[3];
    if (op != OP_JUMP && op != OP_JUMPI)
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    // PUSH2: stack overflow, then its 3 gas.
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
        return fail(EVMC_OUT_OF_GAS);
    // The barrier keeps GCC's bswap pass from treating the two bytes as a big-endian halfword
    // load, which it expands into 7 instructions on rv32 (no rev8); this is 3.
    auto dst = static_cast<uint32_t>(pos.code_it[1]);
    asm("" : "+r"(dst));
    dst = dst << 8 | pos.code_it[2];
    const auto& analysis = *state.analysis.baseline;
    if (op == OP_JUMP)
    {
        // JUMP: the pushed item is its operand, so only the 8 gas can fail.
        if (INTX_UNLIKELY(!deduct_gas(gas, 8)))
            return fail(EVMC_OUT_OF_GAS);
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = skip_landing_jumpdest(&analysis.code()[dst], gas, state);
        return true;
    }
    // JUMPI: underflow unless the condition is under the pushed destination, then 10 gas.
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 10)))
        return fail(EVMC_OUT_OF_GAS);
    // stack_end is one past the top item: the condition is the top item, under the pushed
    // destination.
    const auto* const cw = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    const bool taken = (cw[0] | cw[1] | cw[2] | cw[3] | cw[4] | cw[5] | cw[6] | cw[7]) != 0;
    pos.stack_end -= 1;  // One pushed, two popped.
    if (taken)
    {
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = skip_landing_jumpdest(&analysis.code()[dst], gas, state);
    }
    else
        pos.code_it += 4;
    return true;
}
/// ISZERO or EQ, then PUSH2 and JUMPI: the conditional branches Solidity emits for `if` and for
/// comparisons (5.3M and 3.0M per 200 mainnet blocks). The comparison result is never written to
/// the stack and read back; it decides the jump directly. Checks run in the separate
/// instructions' order: the comparison's underflow and 3 gas, PUSH2's overflow (EQ popped an
/// item, so it cannot overflow) and 3 gas, JUMPI's 10 gas (its two operands are there).
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_push2_jumpi(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    static_assert(Op == OP_ISZERO || Op == OP_EQ);
    constexpr int required = Op == OP_EQ ? 2 : 1;
    if (pos.code_it[1] != OP_PUSH2 || pos.code_it[4] != OP_JUMPI)
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= stack_bottom + (required - 1)))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
        return fail(EVMC_OUT_OF_GAS);
    if constexpr (Op == OP_ISZERO)
    {
        if (INTX_UNLIKELY(pos.stack_end == stack_limit))
            return fail(EVMC_STACK_OVERFLOW);
    }
    if (INTX_UNLIKELY(!deduct_gas(gas, 3 + 10)))
        return fail(EVMC_OUT_OF_GAS);
    const auto* const a = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    bool taken;
    if constexpr (Op == OP_ISZERO)
        taken = (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0;
    else
    {
        const auto* const b = reinterpret_cast<const uint32_t*>(pos.stack_end - 2);
        taken = ((a[0] ^ b[0]) | (a[1] ^ b[1]) | (a[2] ^ b[2]) | (a[3] ^ b[3]) | (a[4] ^ b[4]) |
                    (a[5] ^ b[5]) | (a[6] ^ b[6]) | (a[7] ^ b[7])) == 0;
    }
    pos.stack_end -= required;  // The comparison leaves one, PUSH2 one more, JUMPI takes two.
    if (taken)
    {
        auto dst = static_cast<uint32_t>(pos.code_it[2]);
        asm("" : "+r"(dst));  // See fused_push2_jump().
        dst = dst << 8 | pos.code_it[3];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = skip_landing_jumpdest(&analysis.code()[dst], gas, state);
    }
    else
        pos.code_it += 5;
    return true;
}

/// DUP1 PUSH4 selector EQ PUSH2 tag JUMPI: one test of Solidity's function dispatcher (2.3M per
/// 200 mainnet blocks, a chain of them per external call). The selector is compared against the
/// top item in place and the result decides the jump; the stack ends as it began. Checks in the
/// separate instructions' order: DUP1's underflow and overflow and 3 gas, PUSH4's overflow and
/// its 3 gas, then EQ's 3, PUSH2's 3 (it cannot overflow: EQ popped one) and JUMPI's 10.
[[gnu::always_inline]] inline bool fused_selector_test(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto* const c = pos.code_it;
    if (c[1] != OP_PUSH4)
        return false;
    if (c[6] != OP_EQ || c[7] != OP_PUSH2 || c[10] != OP_JUMPI)
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
        return fail(EVMC_OUT_OF_GAS);
    if (INTX_UNLIKELY(pos.stack_end + 1 == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3 + 3 + 3 + 10)))
        return fail(EVMC_OUT_OF_GAS);
    // The 4 immediate bytes, big-endian, built with the barrier of push_data_word().
    uint32_t sel = c[2];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[3];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[4];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[5];
    const auto* const w = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    const bool taken =
        ((w[0] ^ sel) | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0;
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[8]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[9];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = skip_landing_jumpdest(&analysis.code()[dst], gas, state);
    }
    else
        pos.code_it += 11;
    return true;
}

/// a < b on the 32-bit words of two 256-bit values, most significant word first.
[[gnu::always_inline]] inline bool lt256(const uint32_t* a, const uint32_t* b) noexcept
{
#pragma GCC unroll 8
    for (int i = 7; i > 0; --i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return a[0] < b[0];
}

/// LT or GT, then ISZERO PUSH2 JUMPI: how Solidity branches on a comparison (2.4M per 200
/// mainnet blocks, loop conditions and bounds checks). The comparison decides the jump without
/// its result and the inverted result ever reaching the stack. Checks in the separate
/// instructions' order: the comparison's underflow and 3 gas, then ISZERO's 3, PUSH2's 3 (no
/// overflow: the comparison popped one) and JUMPI's 10.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_iszero_push2_jumpi(const uint256* stack_bottom,
    Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    static_assert(Op == OP_LT || Op == OP_GT);
    const auto* const c = pos.code_it;
    if (c[1] != OP_ISZERO || c[2] != OP_PUSH2 || c[5] != OP_JUMPI)
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= stack_bottom + 1))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
        return fail(EVMC_OUT_OF_GAS);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3 + 3 + 10)))
        return fail(EVMC_OUT_OF_GAS);
    const auto* const top = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    const auto* const second = reinterpret_cast<const uint32_t*>(pos.stack_end - 2);
    // LT leaves top < second, GT leaves second < top; ISZERO inverts; JUMPI jumps on non-zero.
    const bool taken = Op == OP_LT ? !lt256(top, second) : !lt256(second, top);
    pos.stack_end -= 2;  // The comparison leaves one of two, PUSH2 one more, JUMPI takes two.
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[3]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[4];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = skip_landing_jumpdest(&analysis.code()[dst], gas, state);
    }
    else
        pos.code_it += 6;
    return true;
}
#endif

/// A helper to invoke the instruction implementation of the given opcode Op.
template <Opcode Op, bool TracingEnabled>
[[release_inline]] inline Position invoke(const CostTable& cost_table, const uint256* stack_bottom,
    const uint256* stack_limit, Position pos, int64_t& gas, ExecutionState& state) noexcept
{
#if EVMONE_RV32_DISPATCH
    if constexpr (Op == OP_PUSH2)
    {
        if (fused_push2_jump(stack_bottom, stack_limit, pos, gas, state))
            return pos;
    }
    else if constexpr (Op == OP_ISZERO || Op == OP_EQ)
    {
        if (fused_cmp_push2_jumpi<Op>(stack_bottom, stack_limit, pos, gas, state))
            return pos;
    }
    else if constexpr (Op == OP_DUP1)
    {
        if (fused_selector_test(stack_bottom, stack_limit, pos, gas, state))
            return pos;
    }
    else if constexpr (Op == OP_LT || Op == OP_GT)
    {
        if (fused_cmp_iszero_push2_jumpi<Op>(stack_bottom, pos, gas, state))
            return pos;
    }
#endif
    // auto starting_gas = gas;
    const auto status =
        check_requirements<Op>(cost_table, gas, pos.stack_end, stack_bottom, stack_limit);
    if (status != EVMC_SUCCESS)
    {
        // if constexpr (TracingEnabled)
        // {
        //     if (status == EVMC_OUT_OF_GAS)
        //     {
        //         switch (Op)
        //         {
        //         case OP_CALL:
        //         case OP_CALLCODE:
        //         case OP_STATICCALL:
        //         case OP_DELEGATECALL:
        //         case OP_CREATE:
        //         case OP_CREATE2:
        //         case OP_KECCAK256:
        //         case OP_MSTORE:
        //             break;

        //         default:
        //             invoke(instr::core::impl<Op>, pos, starting_gas, state);
        //             break;
        //         }
        //     }
        // }
        state.status = status;
        return {nullptr, pos.stack_end};
    }
    auto new_pos = invoke(instr::core::impl<Op>, pos, gas, state);
#if EVMONE_RV32_DISPATCH
    if constexpr (Op == OP_JUMP)
    {
        if (new_pos != nullptr)  // A taken jump lands on a JUMPDEST.
            new_pos = skip_landing_jumpdest(new_pos, gas, state);
    }
    else if constexpr (Op == OP_JUMPI)
    {
        // Taken or not: a JUMPDEST at the next position is executed the same way either way.
        if (new_pos != nullptr && *new_pos == OP_JUMPDEST)
            new_pos = skip_landing_jumpdest(new_pos, gas, state);
    }
#endif
    const auto new_stack_top = pos.stack_end + instr::traits[Op].stack_height_change;
    return {new_pos, new_stack_top};
}


template <bool TracingEnabled>
int64_t dispatch(const CostTable& cost_table, ExecutionState& state, int64_t gas,
    const uint8_t* code, Tracer* tracer = nullptr) noexcept
{
    const auto stack_bottom = state.stack_space.bottom();
    const auto stack_limit = stack_limit_of(stack_bottom);

    // Code iterator and stack top pointer for interpreter loop.
    Position position{code, stack_bottom};

    while (true)  // Guaranteed to terminate because padded code ends with STOP.
    {
        if constexpr (TracingEnabled)
        {
            const auto offset = static_cast<uint32_t>(position.code_it - code);
            const auto stack_height = static_cast<int>(position.stack_end - stack_bottom);
            if (offset < state.original_code.size())  // Skip STOP from code padding.
            {
                tracer->notify_instruction_start(
                    offset, position.stack_end - 1, stack_height, gas, state);
            }
        }

        const auto op = *position.code_it;
        switch (op)
        {
#define ON_OPCODE(OPCODE)                                                                       \
    case OPCODE:                                                                                \
        ASM_COMMENT(OPCODE);                                                                    \
        if (const auto next =                                                                   \
                invoke<OPCODE, TracingEnabled>(                                                 \
                    cost_table, stack_bottom, stack_limit, position, gas, state);               \
            next.code_it == nullptr)                                                            \
        {                                                                                       \
            return gas;                                                                         \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            /* Update current position only when no error,                                      \
               this improves compiler optimization. */                                          \
            position = next;                                                                    \
        }                                                                                       \
        break;

            MAP_OPCODES
#undef ON_OPCODE

        default:
            state.status = EVMC_UNDEFINED_INSTRUCTION;
            return gas;
        }
    }
    intx::unreachable();
}

#if EVMONE_CGOTO_SUPPORTED
int64_t dispatch_cgoto(
    const CostTable& cost_table, ExecutionState& state, int64_t gas, const uint8_t* code) noexcept
{
#pragma GCC diagnostic ignored "-Wpedantic"

    static constexpr void* cgoto_table[] = {
#define ON_OPCODE(OPCODE) &&TARGET_##OPCODE,
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED(_) &&TARGET_OP_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED ON_OPCODE_UNDEFINED_DEFAULT
    };
    // static_assert(std::size(cgoto_table) == 256);

    const auto stack_bottom = state.stack_space.bottom();
    const auto stack_limit = stack_limit_of(stack_bottom);

    // Code iterator and stack top pointer for interpreter loop.
    Position position{code, stack_bottom};

    goto* cgoto_table[*position.code_it];

#define ON_OPCODE(OPCODE)                                                                        \
    TARGET_##OPCODE : ASM_COMMENT(OPCODE);                                                       \
    if (const auto next =                                                                        \
            invoke<OPCODE, false>(cost_table, stack_bottom, stack_limit, position, gas, state);  \
        next.code_it == nullptr)                                                                 \
    {                                                                                            \
        return gas;                                                                              \
    }                                                                                            \
    else                                                                                         \
    {                                                                                            \
        /* Update current position only when no error,                                           \
           this improves compiler optimization. */                                               \
        position = next;                                                                         \
    }                                                                                            \
    goto* cgoto_table[*position.code_it];

    MAP_OPCODES
#undef ON_OPCODE

TARGET_OP_UNDEFINED:
    state.status = EVMC_UNDEFINED_INSTRUCTION;
    return gas;
}
#endif
}  // namespace

evmc_result execute(VM& vm, const evmc_host_interface& host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message& msg, const CodeAnalysis& analysis) noexcept
{
    const auto code = analysis.code();
    const auto code_begin = code.data();
    auto gas = msg.gas;

    auto& state = vm.get_execution_state(static_cast<size_t>(msg.depth));
    state.reset(msg, rev, host, ctx, code);

    state.analysis.baseline = &analysis;  // Assign code analysis for instruction implementations.

    const auto& cost_table = get_baseline_cost_table(state.rev);

    auto* tracer = vm.get_tracer();
    if (INTX_UNLIKELY(tracer != nullptr))
    {
        tracer->notify_execution_start(state.rev, *state.msg, code);
        gas = dispatch<true>(cost_table, state, gas, code_begin, tracer);
    }
    else
    {
#if EVMONE_CGOTO_SUPPORTED
        if (vm.cgoto)
            gas = dispatch_cgoto(cost_table, state, gas, code_begin);
        else
#endif
            gas = dispatch<false>(cost_table, state, gas, code_begin);
    }

    const auto result = make_execution_result(state, gas);

    if (INTX_UNLIKELY(tracer != nullptr))
        tracer->notify_execution_end(result);

    return result;
}

evmc_result execute(evmc_vm* c_vm, const evmc_host_interface* host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message* msg, const uint8_t* code, size_t code_size) noexcept
{
    auto vm = static_cast<VM*>(c_vm);
    const bytes_view container{code, code_size};

    const auto code_analysis = analyze(container);
    return execute(*vm, *host, ctx, rev, *msg, code_analysis);
    // return evmc_result{EVMC_SUCCESS, msg->gas};
}
}  // namespace evmone::baseline
