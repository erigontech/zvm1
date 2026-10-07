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

/// gas_left -= cost when the low word stays non-negative, with nothing changed otherwise. The
/// whole static gas of a fused sequence in one test: if it passes, each separate charge would
/// have passed. A false return may be a false negative (a low word of 2^31 or more); the caller
/// then takes the exact sequential path.
[[gnu::always_inline]] inline bool charge_all(int64_t& gas_left, uint32_t cost) noexcept
{
    const auto g = static_cast<uint64_t>(gas_left);
    auto lo = static_cast<uint32_t>(g) - cost;
    asm("" : "+r"(lo));
    if (static_cast<int32_t>(lo) < 0) [[unlikely]]
        return false;
    gas_left = static_cast<int64_t>((g & 0xffffffff00000000) | lo);
    return true;
}

/// Gives back the landing JUMPDEST's 1 that charge_all() took for a jump not taken. The low word
/// is below 2^31, so adding 1 cannot carry.
[[gnu::always_inline]] inline void refund_landing(int64_t& gas_left) noexcept
{
    const auto g = static_cast<uint64_t>(gas_left);
    gas_left = static_cast<int64_t>(
        (g & 0xffffffff00000000) | static_cast<uint32_t>(static_cast<uint32_t>(g) + 1));
}

/// x != 0, word 0 first. A non-zero branch condition nearly always has a non-zero low word (a
/// comparison result, a flag, a count or an address), so the other 7 words are read only when it is
/// zero.
[[gnu::always_inline]] inline bool nonzero256(const uint32_t* x) noexcept
{
    if (x[0] != 0) [[likely]]
        return true;
    return (x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7]) != 0;
}

/// x == 0, word 0 first, as nonzero256(). No branch hint: ISZERO's operand is zero more often than
/// not, and a hint would move the jump, the common outcome, out of line.
[[gnu::always_inline]] inline bool zero256(const uint32_t* x) noexcept
{
    if (x[0] != 0)
        return false;
    return (x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7]) == 0;
}

/// a == b, word 0 first: values that differ nearly always differ in their low words.
[[gnu::always_inline]] inline bool eq256(const uint32_t* a, const uint32_t* b) noexcept
{
    if (a[0] != b[0])
        return false;
    return ((a[1] ^ b[1]) | (a[2] ^ b[2]) | (a[3] ^ b[3]) | (a[4] ^ b[4]) | (a[5] ^ b[5]) |
               (a[6] ^ b[6]) | (a[7] ^ b[7])) == 0;
}

/// w == sel for a 32-bit sel, word 0 first: a function dispatcher compares one selector against
/// many, so the low word nearly always decides.
[[gnu::always_inline]] inline bool eq256_u32(const uint32_t* w, uint32_t sel) noexcept
{
    if (w[0] != sel) [[likely]]
        return false;
    return (w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0;
}

/// PUSH2 followed by JUMP or JUMPI, which is how nearly every jump is written (99.6% of them on
/// mainnet). The destination is the immediate, so it is never stored to the stack and read back
/// and its high words need no zero check, and the landing JUMPDEST is folded in. The checks run
/// in the order the separate instructions would run them, so a failure gets the same status.
[[gnu::always_inline]] inline bool fused_push2_jump(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    const auto op = pos.code_it[3];
    if (op == OP_JUMP)
    {
        // PUSH2 3, JUMP 8, the landing JUMPDEST 1.
        if (!charge_all(gas, 3 + 8 + 1)) [[unlikely]]
            return false;
        if (INTX_UNLIKELY(pos.stack_end == stack_limit))
            return fail(EVMC_STACK_OVERFLOW);
        auto dst = static_cast<uint32_t>(pos.code_it[1]);
        asm("" : "+r"(dst));
        dst = dst << 8 | pos.code_it[2];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = &analysis.code()[dst] + 1;
        return true;
    }
    if (op != OP_JUMPI)
        return false;
    // PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 10 + 1)) [[unlikely]]
        return false;
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    pos.stack_end -= 1;  // One pushed, two popped.
    asm("" : "+r"(pos.stack_end));  // Address the popped words from the new stack_end only.
    // The condition is the popped top item, under the pushed destination.
    const bool taken = nonzero256(reinterpret_cast<const uint32_t*>(pos.stack_end));
    if (taken)
    {
        auto dst = static_cast<uint32_t>(pos.code_it[1]);
        asm("" : "+r"(dst));
        dst = dst << 8 | pos.code_it[2];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = &analysis.code()[dst] + 1;
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 4;
    }
    return true;
}
/// ISZERO or EQ, then PUSH2 and JUMPI: the conditional branches Solidity emits for `if` and for
/// comparisons (5.3M and 3.0M per 200 mainnet blocks). The comparison result is never written to
/// the stack and read back; it decides the jump directly. Checks run in the separate
/// instructions' order: the comparison's underflow and 3 gas, PUSH2's overflow (EQ popped an
/// item, so it cannot overflow) and 3 gas, JUMPI's 10 gas (its two operands are there).
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_push2_jumpi_seq(const uint256* stack_bottom,
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

/// The fast form of fused_cmp_push2_jumpi_seq(): the whole static gas in one test.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_push2_jumpi(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    static_assert(Op == OP_ISZERO || Op == OP_EQ);
    constexpr int required = Op == OP_EQ ? 2 : 1;
    if (pos.code_it[1] != OP_PUSH2 || pos.code_it[4] != OP_JUMPI)
        return false;
    // The comparison 3, PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 3 + 10 + 1)) [[unlikely]]
    {
        if constexpr (Op == OP_ISZERO)
            return false;
        else
            return fused_cmp_push2_jumpi_seq<Op>(stack_bottom, stack_limit, pos, gas, state);
    }
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= stack_bottom + (required - 1)))
        return fail(EVMC_STACK_UNDERFLOW);
    if constexpr (Op == OP_ISZERO)
    {
        if (INTX_UNLIKELY(pos.stack_end == stack_limit))
            return fail(EVMC_STACK_OVERFLOW);
    }
    pos.stack_end -= required;  // The comparison leaves one, PUSH2 one more, JUMPI takes two.
    asm("" : "+r"(pos.stack_end));  // Address the popped words from the new stack_end only.
    // The operands are the popped items: the top one a and, for EQ, b under it.
    const auto* const a = reinterpret_cast<const uint32_t*>(pos.stack_end + (required - 1));
    bool taken;
    if constexpr (Op == OP_ISZERO)
        taken = zero256(a);
    else
        taken = eq256(a, reinterpret_cast<const uint32_t*>(pos.stack_end));
    if (taken)
    {
        auto dst = static_cast<uint32_t>(pos.code_it[2]);
        asm("" : "+r"(dst));
        dst = dst << 8 | pos.code_it[3];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = &analysis.code()[dst] + 1;
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 5;
    }
    return true;
}

/// DUP1 PUSH4 selector EQ PUSH2 tag JUMPI: one test of Solidity's function dispatcher (2.3M per
/// 200 mainnet blocks, a chain of them per external call). The selector is compared against the
/// top item in place and the result decides the jump; the stack ends as it began. Checks in the
/// separate instructions' order: DUP1's underflow and overflow and 3 gas, PUSH4's overflow and
/// its 3 gas, then EQ's 3, PUSH2's 3 (it cannot overflow: EQ popped one) and JUMPI's 10.
[[gnu::always_inline]] inline bool fused_selector_test_seq(const uint256* stack_bottom,
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

/// The fast form of fused_selector_test_seq(): the whole static gas in one test.
[[gnu::always_inline]] inline bool fused_selector_test(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto* const c = pos.code_it;
    if (c[1] != OP_PUSH4)
        return false;
    if (c[6] != OP_EQ || c[7] != OP_PUSH2 || c[10] != OP_JUMPI)
        return false;
    // DUP1 3, PUSH4 3, EQ 3, PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 3 + 3 + 3 + 10 + 1)) [[unlikely]]
        return fused_selector_test_seq(stack_bottom, stack_limit, pos, gas, state);
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end + 1 == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    uint32_t sel = c[2];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[3];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[4];
    asm("" : "+r"(sel));
    sel = sel << 8 | c[5];
    const auto* const w = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    const bool taken = eq256_u32(w, sel);
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[8]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[9];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = &analysis.code()[dst] + 1;
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 11;
    }
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
[[gnu::always_inline]] inline bool fused_cmp_iszero_push2_jumpi_seq(const uint256* stack_bottom,
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
/// The fast form of fused_cmp_iszero_push2_jumpi_seq(): the whole static gas in one test.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_iszero_push2_jumpi(const uint256* stack_bottom,
    Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    static_assert(Op == OP_LT || Op == OP_GT);
    const auto* const c = pos.code_it;
    if (c[1] != OP_ISZERO || c[2] != OP_PUSH2 || c[5] != OP_JUMPI)
        return false;
    // The comparison 3, ISZERO 3, PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 3 + 3 + 10 + 1)) [[unlikely]]
        return fused_cmp_iszero_push2_jumpi_seq<Op>(stack_bottom, pos, gas, state);
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= stack_bottom + 1))
        return fail(EVMC_STACK_UNDERFLOW);
    const auto* const top = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
    const auto* const second = reinterpret_cast<const uint32_t*>(pos.stack_end - 2);
    const bool taken = Op == OP_LT ? !lt256(top, second) : !lt256(second, top);
    pos.stack_end -= 2;
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[3]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[4];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = &analysis.code()[dst] + 1;
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 6;
    }
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
    else if constexpr (Op == OP_JUMP)
    {
        if (INTX_UNLIKELY(pos.stack_end <= stack_bottom))
        {
            state.status = EVMC_STACK_UNDERFLOW;
            return {nullptr, pos.stack_end};
        }
        // JUMP 8 and the landing JUMPDEST 1.
        if (charge_all(gas, 8 + 1)) [[likely]]
        {
            const auto* const w = reinterpret_cast<const uint32_t*>(pos.stack_end - 1);
            const uint32_t dst = w[0];
            const auto& analysis = *state.analysis.baseline;
            if (INTX_UNLIKELY((w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) != 0 ||
                              !analysis.check_jumpdest(dst)))
            {
                state.status = EVMC_BAD_JUMP_DESTINATION;
                return {nullptr, pos.stack_end};
            }
            return {&analysis.code()[dst] + 1, pos.stack_end - 1};
        }
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

#if EVMONE_RV32_DISPATCH
/// PUSH1 and its successor.
///
/// PUSH1 is the most executed instruction (29.3M per 200 mainnet blocks) and its successor is
/// known at the point where the handler would push: the dispatch of that successor is the same
/// 5 instructions whichever table it goes through. The PUSH1 handler therefore makes its own
/// checks and dispatches the successor through push1_table, whose entries make the push and
/// jump to the successor's handler (one extra instruction) or run the pair fused: SHL and SHR by
/// an immediate (3.0M), MLOAD and MSTORE at an immediate offset (2.8M) and the mask idiom
/// PUSH1 PUSH1 SHL SUB (1.35M). The fused entries read the immediate from the code instead of
/// writing it to the stack and reading it back, skip its 224-bit zero checks and one dispatch.
/// Checks keep the separate instructions' order, so a failure stops with the same status.

/// The push itself: the immediate is the byte before the successor.
[[gnu::always_inline]] inline void push1_commit(Position& pos) noexcept
{
    auto* const w = reinterpret_cast<uint32_t*>(pos.stack_end);
    w[0] = pos.code_it[-1];
    w[1] = 0;
    w[2] = 0;
    w[3] = 0;
    w[4] = 0;
    w[5] = 0;
    w[6] = 0;
    w[7] = 0;
    pos.stack_end += 1;
}

/// The successors that run fused with PUSH1 through push1_then().
constexpr bool push1_fuses(Opcode op) noexcept
{
    return op == OP_SHL || op == OP_SHR || op == OP_MLOAD || op == OP_MSTORE;
}

/// Word I of x <<= 32 * WS + bs, in place. Destinations are written from the most significant
/// word down, so every source word is read before it is overwritten. (v >> 1) >> rs is
/// v >> (32 - bs), and 0 for bs == 0 where a single shift would be by 32.
template <unsigned WS, unsigned I>
[[gnu::always_inline]] inline void shl_word(uint32_t* x, unsigned bs, unsigned rs) noexcept
{
    if constexpr (I < WS)
        x[I] = 0;
    else if constexpr (I == WS)
        x[I] = x[0] << bs;
    else
        x[I] = (x[I - WS] << bs) | ((x[I - WS - 1] >> 1) >> rs);
}

template <unsigned WS>
[[gnu::always_inline]] inline void shl_words(uint32_t* x, unsigned bs) noexcept
{
    const unsigned rs = 31 - bs;
    shl_word<WS, 7>(x, bs, rs);
    shl_word<WS, 6>(x, bs, rs);
    shl_word<WS, 5>(x, bs, rs);
    shl_word<WS, 4>(x, bs, rs);
    shl_word<WS, 3>(x, bs, rs);
    shl_word<WS, 2>(x, bs, rs);
    shl_word<WS, 1>(x, bs, rs);
    shl_word<WS, 0>(x, bs, rs);
}

/// Word I of x >>= 32 * WS + bs, in place, written from the least significant word up.
template <unsigned WS, unsigned I>
[[gnu::always_inline]] inline void shr_word(uint32_t* x, unsigned bs, unsigned rs) noexcept
{
    if constexpr (I + WS > 7)
        x[I] = 0;
    else if constexpr (I + WS == 7)
        x[I] = x[7] >> bs;
    else
        x[I] = (x[I + WS] >> bs) | ((x[I + WS + 1] << 1) << rs);
}

template <unsigned WS>
[[gnu::always_inline]] inline void shr_words(uint32_t* x, unsigned bs) noexcept
{
    const unsigned rs = 31 - bs;
    shr_word<WS, 0>(x, bs, rs);
    shr_word<WS, 1>(x, bs, rs);
    shr_word<WS, 2>(x, bs, rs);
    shr_word<WS, 3>(x, bs, rs);
    shr_word<WS, 4>(x, bs, rs);
    shr_word<WS, 5>(x, bs, rs);
    shr_word<WS, 6>(x, bs, rs);
    shr_word<WS, 7>(x, bs, rs);
}

/// The fused PUSH1 + Op, entered after PUSH1's own checks with the push not yet made: the
/// immediate is pos.code_it[-1] and pos.stack_end is still the slot it would take. On false the
/// status is set. On true pos is at the next instruction.
template <Opcode Op>
[[gnu::always_inline]] inline bool push1_then(const CostTable& cost_table,
    const uint256* stack_bottom, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        return false;
    };
    const uint32_t imm = pos.code_it[-1];
    if constexpr (Op == OP_SHL || Op == OP_SHR)
    {
        // Constantinople instructions: undefined before it, which check_requirements() tests
        // first, through the revision's cost table.
        if (INTX_UNLIKELY(cost_table[Op] < 0))
            return fail(EVMC_UNDEFINED_INSTRUCTION);
        // Two operands: the pushed shift and the value under it, then 3 gas.
        if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
            return fail(EVMC_STACK_UNDERFLOW);
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        auto* const x = reinterpret_cast<uint32_t*>(pos.stack_end - 1);
        const unsigned bs = imm & 31;
        switch (imm >> 5)  // The word shift: a byte is below 256, so there is no "all out" case.
        {
#define SHIFT_CASE(WS)                             \
    case WS:                                       \
        if constexpr (Op == OP_SHL)                \
            shl_words<WS>(x, bs);                  \
        else                                       \
            shr_words<WS>(x, bs);                  \
        break;
            SHIFT_CASE(0)
            SHIFT_CASE(1)
            SHIFT_CASE(2)
            SHIFT_CASE(3)
            SHIFT_CASE(4)
            SHIFT_CASE(5)
            SHIFT_CASE(6)
            SHIFT_CASE(7)
#undef SHIFT_CASE
        default:
            intx::unreachable();
        }
        pos.code_it += 1;
        return true;
    }
    else if constexpr (Op == OP_MLOAD || Op == OP_MSTORE)
    {
        if constexpr (Op == OP_MSTORE)  // The value under the pushed offset.
        {
            if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
                return fail(EVMC_STACK_UNDERFLOW);
        }
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        // check_memory() for an offset of one byte: no high words to test.
        auto& memory = state.memory;
        if (imm + 32 > memory.size())
        {
            gas = grow_memory(gas, memory, imm + 32);
            if (gas < 0) [[unlikely]]
                return fail(EVMC_OUT_OF_GAS);
        }
        if constexpr (Op == OP_MLOAD)
        {
#ifdef EVMONE_WORD_LAYOUT
            wl::load_u256(*pos.stack_end, &memory[imm]);  // Into the push's slot.
#elif defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32
            intx::be::unsafe::load_into(*pos.stack_end, &memory[imm]);  // Into the push's slot.
#else
            *pos.stack_end = intx::be::unsafe::load<uint256>(&memory[imm]);
#endif
            pos.stack_end += 1;
        }
        else
        {
#ifdef EVMONE_WORD_LAYOUT
            wl::store_u256(&memory[imm], pos.stack_end[-1]);
#else
            intx::be::unsafe::store(&memory[imm], pos.stack_end[-1]);
#endif
            pos.stack_end -= 1;
        }
        pos.code_it += 1;
        return true;
    }
    else
        return true;  // Not fused; dispatch_cgoto() does not call this for other opcodes.
}

/// PUSH1 a PUSH1 b SHL SUB, the mask idiom (for 2^160 - 1 Solidity emits PUSH1 1 PUSH1 1 PUSH1
/// 0xa0 SHL SUB): the top item x becomes (a << b) - x. Entered after the first PUSH1's checks
/// with pos at the second PUSH1. The shifted constant is built in the first push's free slot and
/// the subtraction is the one BigInt delegation SUB makes anyway.
[[gnu::always_inline]] inline bool push1_shl_sub(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        return false;
    };
    // Second PUSH1: overflow with one item already pushed, 3 gas. SHL has its two operands: 3
    // gas. SUB needs x under the shifted constant: underflow, 3 gas.
    // The caller has charged the 9 gas with charge_all().
    (void)gas;
    if (INTX_UNLIKELY(pos.stack_end + 1 == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    const uint32_t a = pos.code_it[-1];
    const uint32_t b = pos.code_it[1];
    // c = a << b: a byte shifted by b < 256 lands in words b / 32 and b / 32 + 1.
    auto* const c = reinterpret_cast<uint32_t*>(pos.stack_end);
    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
    c[6] = 0;
    c[7] = 0;
    const unsigned ws = b >> 5;
    const unsigned bs = b & 31;
    c[ws] = a << bs;
    if (ws < 7)
        c[ws + 1] = (a >> 1) >> (31 - bs);
    // x = c - x: CSR SUB_AND_NEGATE (0x04) is *x10 = *x11 - *x10, as instr::core::sub() uses it.
#if defined(AIRBENDER) && defined(__riscv)
    register uintptr_t r10 asm("x10") = reinterpret_cast<uintptr_t>(pos.stack_end - 1);
    register uintptr_t r11 asm("x11") = reinterpret_cast<uintptr_t>(c);
    register uint32_t r12 asm("x12") = 0x04;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
#else
    pos.stack_end[-1] = pos.stack_end[0] - pos.stack_end[-1];
#endif
    pos.code_it += 4;
    return true;
}

/// One 256-bit stack slot to another (distinct and 32-byte aligned): one MEMCOPY delegation.
[[gnu::always_inline]] inline void copy_slot(uint256* dst, const uint256* src) noexcept
{
#if defined(AIRBENDER) && defined(__riscv)
    register uintptr_t r10 asm("x10") = reinterpret_cast<uintptr_t>(dst);
    register uintptr_t r11 asm("x11") = reinterpret_cast<uintptr_t>(src);
    register uint32_t r12 asm("x12") = 0x80;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
#else
    *dst = *src;
#endif
}

/// *dst += *src: the one BigInt delegation ADD makes.
[[gnu::always_inline]] inline void add_slot(uint256* dst, const uint256* src) noexcept
{
#if defined(AIRBENDER) && defined(__riscv)
    register uintptr_t r10 asm("x10") = reinterpret_cast<uintptr_t>(dst);
    register uintptr_t r11 asm("x11") = reinterpret_cast<uintptr_t>(src);
    register uint32_t r12 asm("x12") = 0x01;
    asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
#else
    *dst += *src;
#endif
}

constexpr bool swap1_fuses(Opcode op) noexcept
{
    return op == OP_POP || op == OP_JUMP || op == OP_SWAP2 || op == OP_DUP2;
}
constexpr bool swap2_fuses(Opcode op) noexcept
{
    return op == OP_POP || op == OP_SWAP1 || op == OP_ADD;
}

/// SWAP1 and Op, entered after SWAP1's checks with the swap not yet made; pos.code_it is at Op.
template <Opcode Op>
[[gnu::always_inline]] inline bool swap1_then(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        return false;
    };
    auto* const s = pos.stack_end;
    if constexpr (Op == OP_POP)
    {
        if (INTX_UNLIKELY(!deduct_gas(gas, 2)))
            return fail(EVMC_OUT_OF_GAS);
        copy_slot(s - 2, s - 1);
        pos.stack_end = s - 1;
        pos.code_it += 1;
        return true;
    }
    else if constexpr (Op == OP_JUMP)
    {
        if (INTX_UNLIKELY(!deduct_gas(gas, 8)))
            return fail(EVMC_OUT_OF_GAS);
        const auto* const w = reinterpret_cast<const uint32_t*>(s - 2);
        const uint32_t dst = w[0];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY((w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) != 0 ||
                          !analysis.check_jumpdest(dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        copy_slot(s - 2, s - 1);
        pos.stack_end = s - 1;
        if (INTX_UNLIKELY(!deduct_gas(gas, 1)))
            return fail(EVMC_OUT_OF_GAS);
        pos.code_it = &analysis.code()[dst] + 1;
        return true;
    }
    else if constexpr (Op == OP_SWAP2)
    {
        if (INTX_UNLIKELY(s <= stack_bottom + 2))
            return fail(EVMC_STACK_UNDERFLOW);
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        alignas(32) char tmp_raw_[sizeof(uint256)];
        auto* const tmp = reinterpret_cast<uint256*>(tmp_raw_);
        copy_slot(tmp, s - 3);
        copy_slot(s - 3, s - 2);
        copy_slot(s - 2, s - 1);
        copy_slot(s - 1, tmp);
        pos.code_it += 1;
        return true;
    }
    else if constexpr (Op == OP_DUP2)
    {
        if (INTX_UNLIKELY(s == stack_limit))
            return fail(EVMC_STACK_OVERFLOW);
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        copy_slot(s, s - 1);
        copy_slot(s - 1, s - 2);
        copy_slot(s - 2, s);
        pos.stack_end = s + 1;
        pos.code_it += 1;
        return true;
    }
    else
        return true;
}

/// SWAP2 and Op, entered after SWAP2's checks with the swap not yet made; pos.code_it is at Op.
template <Opcode Op>
[[gnu::always_inline]] inline bool swap2_then(Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        return false;
    };
    auto* const s = pos.stack_end;
    if constexpr (Op == OP_POP)
    {
        if (INTX_UNLIKELY(!deduct_gas(gas, 2)))
            return fail(EVMC_OUT_OF_GAS);
        copy_slot(s - 3, s - 1);
        pos.stack_end = s - 1;
    }
    else if constexpr (Op == OP_SWAP1)
    {
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        alignas(32) char tmp_raw_[sizeof(uint256)];
        auto* const tmp = reinterpret_cast<uint256*>(tmp_raw_);
        copy_slot(tmp, s - 1);
        copy_slot(s - 1, s - 2);
        copy_slot(s - 2, s - 3);
        copy_slot(s - 3, tmp);
    }
    else if constexpr (Op == OP_ADD)
    {
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        add_slot(s - 2, s - 3);
        copy_slot(s - 3, s - 1);
        pos.stack_end = s - 1;
    }
    pos.code_it += 1;
    return true;
}
#endif

#if EVMONE_CGOTO_SUPPORTED
int64_t dispatch_cgoto(
    const CostTable& cost_table, ExecutionState& state, int64_t gas, const uint8_t* code) noexcept
{
#pragma GCC diagnostic ignored "-Wpedantic"

#if EVMONE_RV32_DISPATCH
    static constexpr void* tables[] = {
#define ON_OPCODE(OPCODE) &&SWAP2_THEN_##OPCODE,
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED(_) &&SWAP2_THEN_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE(OPCODE) &&SWAP1_THEN_##OPCODE,
#define ON_OPCODE_UNDEFINED(_) &&SWAP1_THEN_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE(OPCODE) &&TARGET_##OPCODE,
#define ON_OPCODE_UNDEFINED(_) &&TARGET_OP_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE(OPCODE) &&PUSH1_THEN_##OPCODE,
#define ON_OPCODE_UNDEFINED(_) &&PUSH1_THEN_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED ON_OPCODE_UNDEFINED_DEFAULT
    };
    static_assert(std::size(tables) == 1024);
    void* const* tbl = &tables[512];
    asm("" : "+r"(tbl));
    // The entry for an opcode: the opcode's slot, hidden from GCC so that the table offset
    // stays the load's immediate instead of being added to the opcode.
    const auto slot = [tbl](unsigned op) noexcept {
        auto* p = tbl + op;
        asm("" : "+r"(p));
        return p;
    };
#define CGOTO(OP) (slot(OP)[0])
#define PUSH1_CGOTO(OP) (slot(OP)[256])
#define SWAP1_CGOTO(OP) (slot(OP)[-256])
#define SWAP2_CGOTO(OP) (slot(OP)[-512])
#else
    static constexpr void* cgoto_table[] = {
#define ON_OPCODE(OPCODE) &&TARGET_##OPCODE,
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED(_) &&TARGET_OP_UNDEFINED,
        MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_UNDEFINED
#define ON_OPCODE_UNDEFINED ON_OPCODE_UNDEFINED_DEFAULT
    };
#define CGOTO(OP) cgoto_table[OP]
#endif

    const auto stack_bottom = state.stack_space.bottom();
    const auto stack_limit = stack_limit_of(stack_bottom);

    // Code iterator and stack top pointer for interpreter loop.
    Position position{code, stack_bottom};

    goto* CGOTO(*position.code_it);

#define ON_OPCODE_INVOKE(OPCODE)                                                                 \
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
    goto* CGOTO(*position.code_it);

#if EVMONE_RV32_DISPATCH
#define ON_OPCODE(OPCODE)                                                                        \
    TARGET_##OPCODE : ASM_COMMENT(OPCODE);                                                       \
    if constexpr (OPCODE == OP_PUSH1)                                                            \
    {                                                                                            \
        /* PUSH1's checks; the push is made by the successor's PUSH1_THEN_ entry. */             \
        if (INTX_UNLIKELY(position.stack_end == stack_limit))                                    \
        {                                                                                        \
            state.status = EVMC_STACK_OVERFLOW;                                                  \
            return gas;                                                                          \
        }                                                                                        \
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))                                                  \
        {                                                                                        \
            state.status = EVMC_OUT_OF_GAS;                                                      \
            return gas;                                                                          \
        }                                                                                        \
        position.code_it += 2;                                                                   \
        goto* PUSH1_CGOTO(*position.code_it);                                                    \
    }                                                                                            \
    else if constexpr (OPCODE == OP_SWAP1 || OPCODE == OP_SWAP2)                                 \
    {                                                                                            \
        /* The swap's checks; the swap is made by the successor's SWAPn_THEN_ entry. */          \
        if (const auto status = check_requirements<OPCODE>(                                      \
                cost_table, gas, position.stack_end, stack_bottom, stack_limit);                 \
            status != EVMC_SUCCESS)                                                              \
        {                                                                                        \
            state.status = status;                                                               \
            return gas;                                                                          \
        }                                                                                        \
        position.code_it += 1;                                                                   \
        if constexpr (OPCODE == OP_SWAP1)                                                        \
            goto* SWAP1_CGOTO(*position.code_it);                                                \
        else                                                                                     \
            goto* SWAP2_CGOTO(*position.code_it);                                                \
    }                                                                                            \
    else                                                                                         \
    {                                                                                            \
        ON_OPCODE_INVOKE(OPCODE)                                                                 \
    }
#else
#define ON_OPCODE(OPCODE)                                                                        \
    TARGET_##OPCODE : ASM_COMMENT(OPCODE);                                                       \
    ON_OPCODE_INVOKE(OPCODE)
#endif

    MAP_OPCODES
#undef ON_OPCODE
#undef ON_OPCODE_INVOKE

TARGET_OP_UNDEFINED:
    state.status = EVMC_UNDEFINED_INSTRUCTION;
    return gas;

#if EVMONE_RV32_DISPATCH
#define ON_OPCODE(OPCODE)                                                                        \
    PUSH1_THEN_##OPCODE : ASM_COMMENT(PUSH1_##OPCODE);                                           \
    if constexpr (push1_fuses(OPCODE))                                                           \
    {                                                                                            \
        if (!push1_then<OPCODE>(cost_table, stack_bottom, position, gas, state))                 \
            return gas;                                                                          \
        goto* CGOTO(*position.code_it);                                                    \
    }                                                                                            \
    else if constexpr (OPCODE == OP_PUSH1)                                                       \
    {                                                                                            \
        if (position.code_it[2] == OP_SHL && position.code_it[3] == OP_SUB &&                    \
            cost_table[OP_SHL] >= 0 && charge_all(gas, 3 + 3 + 3))                               \
        {                                                                                        \
            if (!push1_shl_sub(stack_bottom, stack_limit, position, gas, state))                 \
                return gas;                                                                      \
            goto* CGOTO(*position.code_it);                                                \
        }                                                                                        \
        push1_commit(position);                                                                  \
        goto TARGET_##OPCODE;                                                                    \
    }                                                                                            \
    else                                                                                         \
    {                                                                                            \
        push1_commit(position);                                                                  \
        goto TARGET_##OPCODE;                                                                    \
    }

    MAP_OPCODES
#undef ON_OPCODE

PUSH1_THEN_UNDEFINED:
    push1_commit(position);
    goto TARGET_OP_UNDEFINED;

#define ON_OPCODE(OPCODE)                                                                        \
    SWAP1_THEN_##OPCODE : ASM_COMMENT(SWAP1_##OPCODE);                                           \
    if constexpr (swap1_fuses(OPCODE))                                                           \
    {                                                                                            \
        if (!swap1_then<OPCODE>(stack_bottom, stack_limit, position, gas, state))                \
            return gas;                                                                          \
        goto* CGOTO(*position.code_it);                                                          \
    }                                                                                            \
    else                                                                                         \
    {                                                                                            \
        instr::core::swap<1>(position.stack_end);                                                \
        goto TARGET_##OPCODE;                                                                    \
    }

    MAP_OPCODES
#undef ON_OPCODE

SWAP1_THEN_UNDEFINED:
    instr::core::swap<1>(position.stack_end);
    goto TARGET_OP_UNDEFINED;

#define ON_OPCODE(OPCODE)                                                                        \
    SWAP2_THEN_##OPCODE : ASM_COMMENT(SWAP2_##OPCODE);                                           \
    if constexpr (swap2_fuses(OPCODE))                                                           \
    {                                                                                            \
        if (!swap2_then<OPCODE>(position, gas, state))                                           \
            return gas;                                                                          \
        goto* CGOTO(*position.code_it);                                                          \
    }                                                                                            \
    else                                                                                         \
    {                                                                                            \
        instr::core::swap<2>(position.stack_end);                                                \
        goto TARGET_##OPCODE;                                                                    \
    }

    MAP_OPCODES
#undef ON_OPCODE

SWAP2_THEN_UNDEFINED:
    instr::core::swap<2>(position.stack_end);
    goto TARGET_OP_UNDEFINED;
#endif
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
