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

#include <bit>
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

/// stack_bottom + 1, the bound of the 2-operand underflow check (the stack top must lie above
/// it). That check runs in most handlers, and GCC computes the bound again in each (an addi from
/// stack_bottom); hidden like the limit, it is computed once and kept in a register. The 3-operand
/// bound stack_bottom + 2 stays in a register without this while one is free, and under register
/// pressure GCC can then recompute it at each check rather than spill it.
[[gnu::always_inline]] inline const uint256* stack_floor_of(const uint256* stack_bottom) noexcept
{
    const uint256* floor = stack_bottom + 1;
#if EVMONE_RV32_DISPATCH
    asm("" : "+r"(floor));
#endif
    return floor;
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
/// @param          bottom1       stack_bottom + 1, see stack_floor_of().
/// @param          bottom2       stack_bottom + 2.
/// @return  Status code with information which check has failed
///          or EVMC_SUCCESS if everything is fine.
template <Opcode Op>
inline evmc_status_code check_requirements(const CostTable& cost_table, int64_t& gas_left,
    const uint256* stack_top, const uint256* stack_bottom, const uint256* bottom1,
    const uint256* bottom2, const uint256* stack_limit) noexcept
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
        const uint256* floor;
        if constexpr (min_offset == 1)
            floor = bottom1;
        else if constexpr (min_offset == 2)
            floor = bottom2;
        else
            floor = stack_bottom + min_offset;
        if (INTX_UNLIKELY(stack_top <= floor))
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

#if EVMONE_RV32_DISPATCH
/// What a taken jump reads of the analysis besides the scan bound, kept in registers by
/// dispatch_cgoto(): the code start plus one, where a jump to 0 resumes after its landing
/// JUMPDEST, and the JUMPDEST map. Each saves a load per jump (and code1 an addi), with the
/// analysis pointer loaded for the bound alone.
struct JumpBase
{
    code_iterator code1;
    JumpdestMap map;
};

[[gnu::always_inline]] inline JumpBase jump_base(
    const uint8_t* code, const ExecutionState& state) noexcept
{
    return {code + 1, state.analysis.baseline->jumpdest_map()};
}

/// The position after the landing JUMPDEST at dst, a JUMPDEST checked with jb.map.
[[gnu::always_inline]] inline code_iterator landing(const JumpBase& jb, uint32_t dst) noexcept
{
    const auto target = jb.code1 + dst;
    // A code position is never null. GCC cannot see that through the hidden code1 and would
    // test every jump's result against the failure return.
    if (target == nullptr)
        __builtin_unreachable();
    return target;
}
#else
struct JumpBase
{};

[[gnu::always_inline]] inline JumpBase jump_base(const uint8_t*, const ExecutionState&) noexcept
{
    return {};
}
#endif

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
[[gnu::always_inline]] inline bool nonzero256(const word32* x) noexcept
{
    if (x[0] != 0) [[likely]]
        return true;
    return (x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7]) != 0;
}

/// x == 0, word 0 first, as nonzero256(). No branch hint: ISZERO's operand is zero more often than
/// not, and a hint would move the jump, the common outcome, out of line.
[[gnu::always_inline]] inline bool zero256(const word32* x) noexcept
{
    if (x[0] != 0)
        return false;
    return (x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7]) == 0;
}

/// a == b, word 0 first: values that differ nearly always differ in their low words.
[[gnu::always_inline]] inline bool eq256(const word32* a, const word32* b) noexcept
{
    if (a[0] != b[0])
        return false;
    return ((a[1] ^ b[1]) | (a[2] ^ b[2]) | (a[3] ^ b[3]) | (a[4] ^ b[4]) | (a[5] ^ b[5]) |
               (a[6] ^ b[6]) | (a[7] ^ b[7])) == 0;
}

/// w == sel for a 32-bit sel, word 0 first: a function dispatcher compares one selector against
/// many, so the low word nearly always decides.
[[gnu::always_inline]] inline bool eq256_u32(const word32* w, uint32_t sel) noexcept
{
    if (w[0] != sel) [[likely]]
        return false;
    return (w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0;
}

/// a < b on the 32-bit words of two 256-bit values, most significant word first.
[[gnu::always_inline]] inline bool lt256(const word32* a, const word32* b) noexcept
{
#pragma GCC unroll 8
    for (int i = 7; i > 0; --i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return a[0] < b[0];
}

/// a < b as two's complement 256-bit values. The most significant words compare signed: when the
/// signs differ that decides, and with equal signs the signed order is the unsigned one.
[[gnu::always_inline]] inline bool slt256(const word32* a, const word32* b) noexcept
{
    if (a[7] != b[7])
        return static_cast<int32_t>(a[7]) < static_cast<int32_t>(b[7]);
#pragma GCC unroll 7
    for (int i = 6; i > 0; --i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return a[0] < b[0];
}

/// Whether the comparison Op holds for the top item a and the item b under it: LT is a < b, GT is
/// b < a, SLT and SGT the same on signed values.
template <Opcode Op>
[[gnu::always_inline]] inline bool cmp_holds(const word32* a, const word32* b) noexcept
{
    static_assert(Op == OP_LT || Op == OP_GT || Op == OP_SLT || Op == OP_SGT);
    if constexpr (Op == OP_LT)
        return lt256(a, b);
    else if constexpr (Op == OP_GT)
        return lt256(b, a);
    else if constexpr (Op == OP_SLT)
        return slt256(a, b);
    else
        return slt256(b, a);
}

/// PUSH2 followed by JUMP or JUMPI, which is how nearly every jump is written (99.6% of them on
/// mainnet). The destination is the immediate, so it is never stored to the stack and read back
/// and its high words need no zero check, and the landing JUMPDEST is folded in. The checks run
/// in the order the separate instructions would run them, so a failure gets the same status.
[[gnu::always_inline]] inline bool fused_push2_jump(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state,
    JumpBase jb) noexcept
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
        if (INTX_UNLIKELY(!analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = landing(jb, dst);
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
    const bool taken = nonzero256(reinterpret_cast<const word32*>(pos.stack_end));
    if (taken)
    {
        auto dst = static_cast<uint32_t>(pos.code_it[1]);
        asm("" : "+r"(dst));
        dst = dst << 8 | pos.code_it[2];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = landing(jb, dst);
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
    const uint256* bottom1, const uint256* stack_limit, Position& pos, int64_t& gas,
    ExecutionState& state) noexcept
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
    if (INTX_UNLIKELY(pos.stack_end <= (required == 2 ? bottom1 : stack_bottom)))
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
    const word32* const a = reinterpret_cast<const word32*>(pos.stack_end - 1);
    bool taken;
    if constexpr (Op == OP_ISZERO)
        taken = (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0;
    else
    {
        const word32* const b = reinterpret_cast<const word32*>(pos.stack_end - 2);
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

/// The fast form of fused_cmp_push2_jumpi_seq(): the whole static gas in one test. It also takes
/// LT, GT, SLT and SGT, which jump when the comparison holds (2.3M per 200 mainnet blocks) and
/// whose checks are EQ's. When the test fails, the separate instructions run, the exact
/// reference, except after EQ: unfused, EQ makes a BigInt delegation that its sequential form
/// avoids.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_push2_jumpi(const uint256* stack_bottom,
    const uint256* bottom1, const uint256* stack_limit, Position& pos, int64_t& gas,
    ExecutionState& state, JumpBase jb) noexcept
{
    static_assert(Op == OP_ISZERO || Op == OP_EQ || Op == OP_LT || Op == OP_GT || Op == OP_SLT ||
                  Op == OP_SGT);
    constexpr int required = Op == OP_ISZERO ? 1 : 2;
    if (pos.code_it[1] != OP_PUSH2 || pos.code_it[4] != OP_JUMPI)
        return false;
    // The comparison 3, PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 3 + 10 + 1)) [[unlikely]]
    {
        if constexpr (Op == OP_EQ)
            return fused_cmp_push2_jumpi_seq<Op>(
                stack_bottom, bottom1, stack_limit, pos, gas, state);
        else
            return false;
    }
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= (required == 2 ? bottom1 : stack_bottom)))
        return fail(EVMC_STACK_UNDERFLOW);
    if constexpr (Op == OP_ISZERO)
    {
        if (INTX_UNLIKELY(pos.stack_end == stack_limit))
            return fail(EVMC_STACK_OVERFLOW);
    }
    pos.stack_end -= required;  // The comparison leaves one, PUSH2 one more, JUMPI takes two.
    asm("" : "+r"(pos.stack_end));  // Address the popped words from the new stack_end only.
    // The operands are the popped items: the top one a and, with two operands, b under it.
    const word32* const a = reinterpret_cast<const word32*>(pos.stack_end + (required - 1));
    bool taken;
    if constexpr (Op == OP_ISZERO)
        taken = zero256(a);
    else if constexpr (Op == OP_EQ)
        taken = eq256(a, reinterpret_cast<const word32*>(pos.stack_end));
#if defined(AIRBENDER) && defined(__riscv)
    else if constexpr (Op == OP_LT || Op == OP_GT)
    {
        // The SUB delegation instr::core::lt() and gt() make, in 4 instructions where the word
        // compare takes about 23: x12 = the borrow of *x10 - *x11, top minus second for LT and
        // second minus top for GT. The difference overwrites a popped item. The ISZERO form
        // compares words instead: fused, it has never made a delegation call.
        uint256* const top = pos.stack_end + 1;
        register uintptr_t r10 asm("x10") =
            reinterpret_cast<uintptr_t>(Op == OP_LT ? top : pos.stack_end);
        register uintptr_t r11 asm("x11") =
            reinterpret_cast<uintptr_t>(Op == OP_LT ? pos.stack_end : top);
        register uint32_t r12 asm("x12") = 0x02;
        asm volatile("csrrw x0, 0x7CA, x0" : "+r"(r12) : "r"(r10), "r"(r11) : "memory");
        taken = r12 != 0;
    }
#endif
    else
        taken = cmp_holds<Op>(a, reinterpret_cast<const word32*>(pos.stack_end));
    if (taken)
    {
        auto dst = static_cast<uint32_t>(pos.code_it[2]);
        asm("" : "+r"(dst));
        dst = dst << 8 | pos.code_it[3];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = landing(jb, dst);
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
    const word32* const w = reinterpret_cast<const word32*>(pos.stack_end - 1);
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
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state,
    JumpBase jb) noexcept
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
    const word32* const w = reinterpret_cast<const word32*>(pos.stack_end - 1);
    const bool taken = eq256_u32(w, sel);
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[8]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[9];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = landing(jb, dst);
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 11;
    }
    return true;
}

/// LT, GT, SLT or SGT, then ISZERO PUSH2 JUMPI: how Solidity branches on a comparison (3.0M per
/// 200 mainnet blocks, loop conditions and bounds checks). The comparison decides the jump
/// without its result and the inverted result ever reaching the stack. Checks in the separate
/// instructions' order: the comparison's underflow and 3 gas, then ISZERO's 3, PUSH2's 3 (no
/// overflow: the comparison popped one) and JUMPI's 10.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_iszero_push2_jumpi_seq(const uint256* bottom1,
    Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto* const c = pos.code_it;
    if (c[1] != OP_ISZERO || c[2] != OP_PUSH2 || c[5] != OP_JUMPI)
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= bottom1))
        return fail(EVMC_STACK_UNDERFLOW);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
        return fail(EVMC_OUT_OF_GAS);
    if (INTX_UNLIKELY(!deduct_gas(gas, 3 + 3 + 10)))
        return fail(EVMC_OUT_OF_GAS);
    const word32* const top = reinterpret_cast<const word32*>(pos.stack_end - 1);
    const word32* const second = reinterpret_cast<const word32*>(pos.stack_end - 2);
    // ISZERO inverts the comparison; JUMPI jumps on non-zero.
    const bool taken = !cmp_holds<Op>(top, second);
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
/// The fast form of fused_cmp_iszero_push2_jumpi_seq(): the whole static gas in one test. When
/// the test fails, LT and GT take the sequential form: unfused, they make a BigInt delegation
/// that it avoids. SLT and SGT make none, and their separate instructions are the exact reference.
template <Opcode Op>
[[gnu::always_inline]] inline bool fused_cmp_iszero_push2_jumpi(const uint256* bottom1,
    Position& pos, int64_t& gas, ExecutionState& state, JumpBase jb) noexcept
{
    const auto* const c = pos.code_it;
    if (c[1] != OP_ISZERO || c[2] != OP_PUSH2 || c[5] != OP_JUMPI)
        return false;
    // The comparison 3, ISZERO 3, PUSH2 3, JUMPI 10, the landing JUMPDEST 1.
    if (!charge_all(gas, 3 + 3 + 3 + 10 + 1)) [[unlikely]]
    {
        if constexpr (Op == OP_LT || Op == OP_GT)
            return fused_cmp_iszero_push2_jumpi_seq<Op>(bottom1, pos, gas, state);
        else
            return false;
    }
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end <= bottom1))
        return fail(EVMC_STACK_UNDERFLOW);
    const word32* const top = reinterpret_cast<const word32*>(pos.stack_end - 1);
    const word32* const second = reinterpret_cast<const word32*>(pos.stack_end - 2);
    const bool taken = !cmp_holds<Op>(top, second);
    pos.stack_end -= 2;
    if (taken)
    {
        auto dst = static_cast<uint32_t>(c[3]);
        asm("" : "+r"(dst));
        dst = dst << 8 | c[4];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY(!analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        pos.code_it = landing(jb, dst);
    }
    else
    {
        refund_landing(gas);
        pos.code_it += 6;
    }
    return true;
}

/// Whether the 20 immediate bytes of the PUSH20 at c are all 0xff. The 6 aligned words that
/// cover them are read, and the bytes outside the immediate are forced to 0xff: on a
/// little-endian target the low k bytes of the first word precede it and the rest of the last
/// word follows it. The reads stay within the allocation of the analysis copy (see
/// analyze_legacy()): it is at least 4-aligned, so rounding c + 1 down to 4 stays inside it (on
/// the guest the copy may start up to 31 bytes in, after bytes of fresh heap RAM, which read as
/// zero: the guest's calloc does not clear), and the last word
/// ends at most at c + 24, inside the 33 zero bytes after the code (a truncated PUSH20 sees 0x00
/// bytes and fails the test).
[[gnu::always_inline]] inline bool push20_all_ones(const uint8_t* c) noexcept
{
    static_assert(std::endian::native == std::endian::little);
    const uint8_t* const d = c + 1;
    const auto k = static_cast<unsigned>(reinterpret_cast<uintptr_t>(d) & 3);
    const word32* const w = reinterpret_cast<const word32*>(d - k);
    const uint32_t before = (uint32_t{1} << (8 * k)) - 1;
    return ((w[0] | before) & w[1] & w[2] & w[3] & w[4] & (w[5] | ~before)) == ~uint32_t{0};
}

/// PUSH20 0xff..ff AND: Solidity's address mask (1.09M per 200 mainnet blocks). x & (2^160 - 1)
/// clears the 3 high words of x in place; the mask is never written to the stack, nor read back
/// by a separate AND. Checks in the separate instructions' order: PUSH20's overflow, then AND's
/// underflow (the item under the mask), after both charges in one test. When that test fails the
/// two instructions run separately, the exact reference.
[[gnu::always_inline]] inline bool fused_push20_mask_and(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto* const c = pos.code_it;
    if (c[21] != OP_AND || !push20_all_ones(c))
        return false;
    // PUSH20 3, AND 3.
    if (!charge_all(gas, 3 + 3)) [[unlikely]]
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    word32* const x = reinterpret_cast<word32*>(pos.stack_end - 1);
    x[5] = 0;
    x[6] = 0;
    x[7] = 0;
    pos.code_it += 22;
    return true;
}

/// PUSH4 imm AND (332K per 200 mainnet blocks, nearly all with imm 0xffffffff): x & imm has one
/// word. Checks as in fused_push20_mask_and().
[[gnu::always_inline]] inline bool fused_push4_and(const uint256* stack_bottom,
    const uint256* stack_limit, Position& pos, int64_t& gas, ExecutionState& state) noexcept
{
    const auto* const c = pos.code_it;
    // Three in four PUSH4s are followed by something else: keep their path the straight one.
    if (c[5] != OP_AND) [[likely]]
        return false;
    // PUSH4 3, AND 3.
    if (!charge_all(gas, 3 + 3)) [[unlikely]]
        return false;
    const auto fail = [&](evmc_status_code status) noexcept {
        state.status = status;
        pos.code_it = nullptr;
        return true;
    };
    if (INTX_UNLIKELY(pos.stack_end == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    // The 4 immediate bytes, big-endian, built with the barrier of push_data_word().
    uint32_t imm = c[1];
    asm("" : "+r"(imm));
    imm = imm << 8 | c[2];
    asm("" : "+r"(imm));
    imm = imm << 8 | c[3];
    asm("" : "+r"(imm));
    imm = imm << 8 | c[4];
    word32* const x = reinterpret_cast<word32*>(pos.stack_end - 1);
    x[0] &= imm;
    x[1] = 0;
    x[2] = 0;
    x[3] = 0;
    x[4] = 0;
    x[5] = 0;
    x[6] = 0;
    x[7] = 0;
    pos.code_it += 6;
    return true;
}

/// Word I of x >>= 32 * WS + bs arithmetically, in place, written from the least significant word
/// up, so every source word (I + WS and I + WS + 1) is read before it is overwritten. The words
/// above the shifted top word take the sign, which the caller reads before any store.
/// (v << 1) << rs is v << (32 - bs), and 0 for bs == 0 where a single shift would be by 32.
template <unsigned WS, unsigned I>
[[gnu::always_inline]] inline void sar_word(
    word32* x, unsigned bs, unsigned rs, uint32_t sign) noexcept
{
    if constexpr (I + WS > 7)
        x[I] = sign;
    else if constexpr (I + WS == 7)
        x[I] = static_cast<uint32_t>(static_cast<int32_t>(x[7]) >> bs);
    else
        x[I] = (x[I + WS] >> bs) | ((x[I + WS + 1] << 1) << rs);
}

template <unsigned WS>
[[gnu::always_inline]] inline void sar_words(word32* x, unsigned bs, uint32_t sign) noexcept
{
    const unsigned rs = 31 - bs;
    sar_word<WS, 0>(x, bs, rs, sign);
    sar_word<WS, 1>(x, bs, rs, sign);
    sar_word<WS, 2>(x, bs, rs, sign);
    sar_word<WS, 3>(x, bs, rs, sign);
    sar_word<WS, 4>(x, bs, rs, sign);
    sar_word<WS, 5>(x, bs, rs, sign);
    sar_word<WS, 6>(x, bs, rs, sign);
    sar_word<WS, 7>(x, bs, rs, sign);
}

/// x >>= s arithmetically, for s below 256. instr::core::sar() builds two full 256-bit shifts
/// (about 220 instructions on rv32); moving words takes about 25 to 60
/// instructions from the shift decode through the last store.
[[gnu::always_inline]] inline void sar_below_256(word32* x, unsigned s) noexcept
{
    const auto sign = static_cast<uint32_t>(static_cast<int32_t>(x[7]) >> 31);
    const unsigned bs = s & 31;
    switch ((s >> 5) & 7)  // The word shift; the mask spares the switch its range check.
    {
#define SAR_CASE(WS)                \
    case WS:                        \
        sar_words<WS>(x, bs, sign); \
        break;
        SAR_CASE(0)
        SAR_CASE(1)
        SAR_CASE(2)
        SAR_CASE(3)
        SAR_CASE(4)
        SAR_CASE(5)
        SAR_CASE(6)
        SAR_CASE(7)
#undef SAR_CASE
    default:
        intx::unreachable();
    }
}

/// SAR with its shift y from the stack: x >>= y arithmetically, all sign bits from 256 on.
[[gnu::always_inline]] inline void sar_stack(word32* x, const word32* y) noexcept
{
    if ((y[1] | y[2] | y[3] | y[4] | y[5] | y[6] | y[7]) == 0 && y[0] < 256) [[likely]]
        sar_below_256(x, y[0]);
    else
    {
        const auto sign = static_cast<uint32_t>(static_cast<int32_t>(x[7]) >> 31);
        for (int i = 0; i < 8; ++i)
            x[i] = sign;
    }
}

/// Word I of x <<= 32 * WS + bs, in place. Destinations are written from the most significant
/// word down, so every source word is read before it is overwritten. (v >> 1) >> rs is
/// v >> (32 - bs), and 0 for bs == 0 where a single shift would be by 32.
template <unsigned WS, unsigned I>
[[gnu::always_inline]] inline void shl_word(word32* x, unsigned bs, unsigned rs) noexcept
{
    if constexpr (I < WS)
        x[I] = 0;
    else if constexpr (I == WS)
        x[I] = x[0] << bs;
    else
        x[I] = (x[I - WS] << bs) | ((x[I - WS - 1] >> 1) >> rs);
}

template <unsigned WS>
[[gnu::always_inline]] inline void shl_words(word32* x, unsigned bs) noexcept
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
[[gnu::always_inline]] inline void shr_word(word32* x, unsigned bs, unsigned rs) noexcept
{
    if constexpr (I + WS > 7)
        x[I] = 0;
    else if constexpr (I + WS == 7)
        x[I] = x[7] >> bs;
    else
        x[I] = (x[I + WS] >> bs) | ((x[I + WS + 1] << 1) << rs);
}

template <unsigned WS>
[[gnu::always_inline]] inline void shr_words(word32* x, unsigned bs) noexcept
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

/// x <<= s or x >>= s in place, for a shift s below 256. The word shift s >> 5 selects an unrolled
/// body, in which each destination word is one or two shifts and an or. Shared by the shifts of
/// a stack operand and of a PUSH1 immediate.
template <Opcode Op>
[[gnu::always_inline]] inline void shift_words(word32* x, uint32_t s) noexcept
{
    const unsigned bs = s & 31;
    switch (s >> 5)  // Below 256, so there is no "all out" case.
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
}

/// SHL and SHR with both operands on the stack: the value under the shift, the result in the
/// value's slot. The generic implementation, intx's operator<<=(uint), takes the shift by value
/// (8 dead stores to the C stack) and builds the shift from uint128 halves with data-dependent
/// branches: a shift below 128 costs about 45 more instructions than the word bodies. A shift
/// of 256 or more, in any of the 8 words, gives 0, so the high 24 bits of word 0 and all of the
/// other words are ORed rather than compared one by one.
template <Opcode Op>
[[gnu::always_inline]] inline void shift_by_stack(uint256* stack_end) noexcept
{
    const word32* const s = reinterpret_cast<const word32*>(stack_end - 1);
    word32* const x = reinterpret_cast<word32*>(stack_end - 2);
    if (INTX_UNLIKELY(((s[0] >> 8) | s[1] | s[2] | s[3] | s[4] | s[5] | s[6] | s[7]) != 0))
    {
        for (unsigned i = 0; i < 8; ++i)
            x[i] = 0;
    }
    else
        shift_words<Op>(x, s[0]);
}
#endif

/// A helper to invoke the instruction implementation of the given opcode Op.
template <Opcode Op, bool TracingEnabled>
[[release_inline]] inline Position invoke(const CostTable& cost_table, const uint256* stack_bottom,
    const uint256* bottom1, const uint256* bottom2, const uint256* stack_limit, Position pos,
    int64_t& gas, ExecutionState& state, [[maybe_unused]] JumpBase jb) noexcept
{
#if EVMONE_RV32_DISPATCH
    if constexpr (Op == OP_PUSH2)
    {
        if (fused_push2_jump(stack_bottom, stack_limit, pos, gas, state, jb))
            return pos;
    }
    else if constexpr (Op == OP_ISZERO || Op == OP_EQ)
    {
        if (fused_cmp_push2_jumpi<Op>(stack_bottom, bottom1, stack_limit, pos, gas, state, jb))
            return pos;
    }
    else if constexpr (Op == OP_DUP1)
    {
        if (fused_selector_test(stack_bottom, stack_limit, pos, gas, state, jb))
            return pos;
    }
    else if constexpr (Op == OP_LT || Op == OP_GT || Op == OP_SLT || Op == OP_SGT)
    {
        // ISZERO first: it follows each of them more often than PUSH2 does, and on a partial
        // match GCC skips the PUSH2 test, the successor being known to be ISZERO.
        if (fused_cmp_iszero_push2_jumpi<Op>(bottom1, pos, gas, state, jb))
            return pos;
        if (fused_cmp_push2_jumpi<Op>(stack_bottom, bottom1, stack_limit, pos, gas, state, jb))
            return pos;
    }
    else if constexpr (Op == OP_PUSH20)
    {
        if (fused_push20_mask_and(stack_bottom, stack_limit, pos, gas, state))
            return pos;
    }
    else if constexpr (Op == OP_PUSH4)
    {
        if (fused_push4_and(stack_bottom, stack_limit, pos, gas, state))
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
            const word32* const w = reinterpret_cast<const word32*>(pos.stack_end - 1);
            const uint32_t dst = w[0];
            const auto& analysis = *state.analysis.baseline;
            if (INTX_UNLIKELY((w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) != 0 ||
                              !analysis.check_jumpdest_in(jb.map, dst)))
            {
                state.status = EVMC_BAD_JUMP_DESTINATION;
                return {nullptr, pos.stack_end};
            }
            return {landing(jb, dst), pos.stack_end - 1};
        }
    }
#endif
    // auto starting_gas = gas;
    const auto status = check_requirements<Op>(
        cost_table, gas, pos.stack_end, stack_bottom, bottom1, bottom2, stack_limit);
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
#if EVMONE_RV32_DISPATCH
    if constexpr (Op == OP_SAR)
    {
        // By words, after check_requirements(): the shift is the top item and the value under
        // it takes the result.
        sar_stack(reinterpret_cast<word32*>(pos.stack_end - 2),
            reinterpret_cast<const word32*>(pos.stack_end - 1));
        return {pos.code_it + 1, pos.stack_end - 1};
    }
    if constexpr (Op == OP_SHL || Op == OP_SHR)
    {
        shift_by_stack<Op>(pos.stack_end);
        return {pos.code_it + 1, pos.stack_end - 1};
    }
#endif
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
    const auto jb = jump_base(code, state);
    const auto bottom1 = stack_floor_of(stack_bottom);
    const auto bottom2 = stack_bottom + 2;

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
                invoke<OPCODE, TracingEnabled>(cost_table, stack_bottom, bottom1, bottom2,      \
                    stack_limit, position, gas, state, jb);                                     \
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
/// an immediate (3.0M), SAR by an immediate, MLOAD and MSTORE at an immediate offset (2.8M), ADD,
/// AND, NOT and SWAP1 with an immediate and the mask idiom PUSH1 PUSH1 SHL SUB (1.35M). The fused
/// entries read the immediate from the code instead of writing it to the stack and reading it
/// back, skip its 224-bit zero checks and one dispatch.
/// Checks keep the separate instructions' order, so a failure stops with the same status.

/// The push itself: the immediate is the byte before the successor.
[[gnu::always_inline]] inline void push1_commit(Position& pos) noexcept
{
    word32* const w = reinterpret_cast<word32*>(pos.stack_end);
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
    return op == OP_SHL || op == OP_SHR || op == OP_SAR || op == OP_MLOAD || op == OP_MSTORE ||
           op == OP_ADD || op == OP_NOT || op == OP_SWAP1 || op == OP_AND;
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
    if constexpr (Op == OP_SHL || Op == OP_SHR || Op == OP_SAR)
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
        word32* const x = reinterpret_cast<word32*>(pos.stack_end - 1);
        if constexpr (Op == OP_SAR)
        {
            sar_below_256(x, imm);  // A byte: below 256.
            pos.code_it += 1;
            return true;
        }
        shift_words<Op>(x, imm);
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
            if (Op == OP_MSTORE && imm == 0x40 && memory.size() == 0)
            {
                // Solidity's prologue PUSH1 0x80 PUSH1 0x40 MSTORE as the frame's first memory
                // access (94% of these growths): 0 to 3 words cost 3 * 3 + 3^2 / 512 = 9 gas.
                if (INTX_UNLIKELY(!deduct_gas(gas, 9)))
                    return fail(EVMC_OUT_OF_GAS);
                memory.grow_empty_for_store_at_64();
            }
            else
            {
                gas = grow_memory(gas, memory, imm + 32);
                if (gas < 0) [[unlikely]]
                    return fail(EVMC_OUT_OF_GAS);
            }
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
    else if constexpr (Op == OP_ADD || Op == OP_AND || Op == OP_SWAP1)
    {
        // Two operands: the pushed one and x under it, then 3 gas.
        if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
            return fail(EVMC_STACK_UNDERFLOW);
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        word32* const x = reinterpret_cast<word32*>(pos.stack_end - 1);
        if constexpr (Op == OP_ADD)
        {
            // The carry runs up through the words that wrap to 0, rarely past the first.
            const uint32_t lo = x[0] + imm;
            x[0] = lo;
            if (INTX_UNLIKELY(lo < imm))
            {
                for (unsigned i = 1; i < 8 && ++x[i] == 0; ++i)
                {
                }
            }
        }
        else
        {
            // AND keeps only the immediate's bits of x[0]. SWAP1 moves x up into the push's slot
            // and the immediate takes its place.
            if constexpr (Op == OP_SWAP1)
            {
                copy_slot(pos.stack_end, pos.stack_end - 1);
                pos.stack_end += 1;
                x[0] = imm;
            }
            else
                x[0] &= imm;
            x[1] = 0;
            x[2] = 0;
            x[3] = 0;
            x[4] = 0;
            x[5] = 0;
            x[6] = 0;
            x[7] = 0;
        }
        pos.code_it += 1;
        return true;
    }
    else if constexpr (Op == OP_NOT)
    {
        // The one operand is the pushed item: no underflow. 3 gas.
        if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
            return fail(EVMC_OUT_OF_GAS);
        word32* const w = reinterpret_cast<word32*>(pos.stack_end);
        w[0] = ~imm;
        w[1] = ~uint32_t{0};
        w[2] = ~uint32_t{0};
        w[3] = ~uint32_t{0};
        w[4] = ~uint32_t{0};
        w[5] = ~uint32_t{0};
        w[6] = ~uint32_t{0};
        w[7] = ~uint32_t{0};
        pos.stack_end += 1;
        pos.code_it += 1;
        return true;
    }
    else
    {
        // An opcode in push1_fuses() without a branch here would neither push nor advance.
        static_assert(!push1_fuses(Op));
        return true;  // Not fused; dispatch_cgoto() does not call this for other opcodes.
    }
}

/// PUSH1 a PUSH1 b SHL SUB, the mask idiom (for 2^160 - 1 Solidity emits PUSH1 1 PUSH1 1 PUSH1
/// 0xa0 SHL SUB): the top item x becomes (a << b) - x. Entered after the first PUSH1's checks
/// with pos at the second PUSH1. The shifted constant is built in the first push's free slot and
/// the subtraction is the one BigInt delegation SUB makes anyway. The address mask followed by
/// AND also takes the AND, without either.
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
    if (INTX_UNLIKELY(pos.stack_end + 1 == stack_limit))
        return fail(EVMC_STACK_OVERFLOW);
    if (INTX_UNLIKELY(pos.stack_end == stack_bottom))
        return fail(EVMC_STACK_UNDERFLOW);
    const uint32_t a = pos.code_it[-1];
    const uint32_t b = pos.code_it[1];
    if (pos.code_it[4] == OP_AND && a == 1 && b == 160)
    {
        // With x == 1 the result is 2^160 - 1, and an AND follows: Solidity's address mask
        // (404K per 200 mainnet blocks). The AND clears the 3 high words of the item y under x,
        // after its own underflow test and 3 gas; neither the mask nor the SUB is made. Other
        // operands that give the same mask take the general path.
        const word32* const x = reinterpret_cast<const word32*>(pos.stack_end - 1);
        if (((x[0] ^ 1) | x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7]) == 0)
        {
            if (INTX_UNLIKELY(pos.stack_end - 1 == stack_bottom))
                return fail(EVMC_STACK_UNDERFLOW);
            if (INTX_UNLIKELY(!deduct_gas(gas, 3)))
                return fail(EVMC_OUT_OF_GAS);
            word32* const y = reinterpret_cast<word32*>(pos.stack_end - 2);
            y[5] = 0;
            y[6] = 0;
            y[7] = 0;
            pos.stack_end -= 1;
            pos.code_it += 5;
            return true;
        }
    }
    // c = a << b: a byte shifted by b < 256 lands in words b / 32 and b / 32 + 1.
    word32* const c = reinterpret_cast<word32*>(pos.stack_end);
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
    // Through one pointer: indexing c twice, GCC computes the address of c[ws + 1] apart.
    word32* const cw = c + ws;
    cw[0] = a << bs;
    if (ws < 7)
        cw[1] = (a >> 1) >> (31 - bs);
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
[[gnu::always_inline]] inline bool swap1_then(const uint256* bottom2, const uint256* stack_limit,
    Position& pos, int64_t& gas, ExecutionState& state, [[maybe_unused]] JumpBase jb) noexcept
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
        const word32* const w = reinterpret_cast<const word32*>(s - 2);
        const uint32_t dst = w[0];
        const auto& analysis = *state.analysis.baseline;
        if (INTX_UNLIKELY((w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) != 0 ||
                          !analysis.check_jumpdest_in(jb.map, dst)))
            return fail(EVMC_BAD_JUMP_DESTINATION);
        copy_slot(s - 2, s - 1);
        pos.stack_end = s - 1;
        if (INTX_UNLIKELY(!deduct_gas(gas, 1)))
            return fail(EVMC_OUT_OF_GAS);
        pos.code_it = landing(jb, dst);
        return true;
    }
    else if constexpr (Op == OP_SWAP2)
    {
        if (INTX_UNLIKELY(s <= bottom2))
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
    auto jb = jump_base(code, state);
#if EVMONE_RV32_DISPATCH
    // Hidden so that GCC keeps both in registers instead of reloading them through the analysis
    // on each jump.
    asm("" : "+r"(jb.code1), "+r"(jb.map));
#endif
    const auto bottom1 = stack_floor_of(stack_bottom);
    const auto bottom2 = stack_bottom + 2;

    // Code iterator and stack top pointer for interpreter loop.
    Position position{code, stack_bottom};

    goto* CGOTO(*position.code_it);

#define ON_OPCODE_INVOKE(OPCODE)                                                                 \
    if (const auto next = invoke<OPCODE, false>(                                                 \
            cost_table, stack_bottom, bottom1, bottom2, stack_limit, position, gas, state, jb);  \
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
        if (const auto status = check_requirements<OPCODE>(cost_table, gas,                      \
                position.stack_end, stack_bottom, bottom1, bottom2, stack_limit);                \
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
        if (!swap1_then<OPCODE>(bottom2, stack_limit, position, gas, state, jb))                 \
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

evmc::Result execute(VM& vm, const evmc_host_interface& host, evmc_host_context* ctx,
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

    // Not const: the result is returned by NRVO (evmc::Result cannot be copied).
    auto result = make_execution_result(state, gas);

    if (INTX_UNLIKELY(tracer != nullptr))
        tracer->notify_execution_end(result.raw());

    return result;
}

evmc_result execute(evmc_vm* c_vm, const evmc_host_interface* host, evmc_host_context* ctx,
    evmc_revision rev, const evmc_message* msg, const uint8_t* code, size_t code_size) noexcept
{
    auto vm = static_cast<VM*>(c_vm);
    const bytes_view container{code, code_size};

    const auto code_analysis = analyze(container);
    return execute(*vm, *host, ctx, rev, *msg, code_analysis).release_raw();
    // return evmc_result{EVMC_SUCCESS, msg->gas};
}
}  // namespace evmone::baseline
