// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2019-2020 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0

#include <evmc/evmc.hpp>
#include <evmone/evmone.h>
#include <evmone/vm.hpp>
#include <gtest/gtest.h>
#include <limits>

TEST(evmone, info)
{
    auto vm = evmc::VM{evmc_create_evmone()};
    EXPECT_STREQ(vm.name(), "evmone");
    EXPECT_STREQ(vm.version(), PROJECT_VERSION);
    EXPECT_TRUE(vm.is_abi_compatible());
}

TEST(evmc, result_with_state_gas)
{
    const auto result = evmc::Result{EVMC_SUCCESS, 1, 2, {.left = 3, .spilled = 4}};
    EXPECT_EQ(result.status_code, EVMC_SUCCESS);
    EXPECT_EQ(result.gas_left, 1);
    EXPECT_EQ(result.gas_refund, 2);
    EXPECT_EQ(result.state_gas.left, 3);
    EXPECT_EQ(result.state_gas.spilled, 4);
    EXPECT_EQ(result.output_data, nullptr);
    EXPECT_EQ(result.output_size, 0);

    const auto default_result = evmc::Result{};
    EXPECT_EQ(default_result.state_gas.left, 0);
    EXPECT_EQ(default_result.state_gas.spilled, 0);

    const uint8_t output[] = {0x01};
    const auto output_result =
        evmc::Result{EVMC_REVERT, 1, 0, output, std::size(output), {.left = 5, .spilled = 6}};
    EXPECT_EQ(output_result.state_gas.left, 5);
    EXPECT_EQ(output_result.state_gas.spilled, 6);

    const auto failure_result = evmc::Result{EVMC_OUT_OF_GAS, {.left = 7}};
    EXPECT_EQ(failure_result.status_code, EVMC_OUT_OF_GAS);
    EXPECT_EQ(failure_result.gas_left, 0);
    EXPECT_EQ(failure_result.gas_refund, 0);
    EXPECT_EQ(failure_result.state_gas.left, 7);
    EXPECT_EQ(failure_result.state_gas.spilled, 0);
}

TEST(evmone, set_option_invalid)
{
    auto vm = evmc_create_evmone();
    ASSERT_NE(vm->set_option, nullptr);
    EXPECT_EQ(vm->set_option(vm, "", ""), EVMC_SET_OPTION_INVALID_NAME);
    EXPECT_EQ(vm->set_option(vm, "o", ""), EVMC_SET_OPTION_INVALID_NAME);
    EXPECT_EQ(vm->set_option(vm, "0", ""), EVMC_SET_OPTION_INVALID_NAME);
    vm->destroy(vm);
}

TEST(evmone, set_option_advanced)
{
    auto vm = evmc::VM{evmc_create_evmone()};
    EXPECT_EQ(vm.set_option("advanced", ""), EVMC_SET_OPTION_SUCCESS);

    // This will also enable Advanced.
    EXPECT_EQ(vm.set_option("advanced", "no"), EVMC_SET_OPTION_SUCCESS);
}

TEST(evmone, set_option_cgoto)
{
    evmc::VM vm{evmc_create_evmone()};

#if EVMONE_CGOTO_SUPPORTED
    EXPECT_EQ(vm.set_option("cgoto", ""), EVMC_SET_OPTION_INVALID_VALUE);
    EXPECT_EQ(vm.set_option("cgoto", "yes"), EVMC_SET_OPTION_INVALID_VALUE);
    EXPECT_EQ(vm.set_option("cgoto", "no"), EVMC_SET_OPTION_SUCCESS);
#else
    EXPECT_EQ(vm.set_option("cgoto", "no"), EVMC_SET_OPTION_INVALID_NAME);
#endif
}

TEST(evmone, max_memory_size)
{
    using evmone::VM;

    // The memory of w words costs 3w + w²/512 gas.
    const auto cost = [](uint64_t words) { return 3 * words + words * words / 512; };

    // At least a page: the least capacity of Memory.
    EXPECT_EQ(VM::max_memory_size(-1), evmone::Memory::page_size);
    EXPECT_EQ(VM::max_memory_size(0), evmone::Memory::page_size);
    EXPECT_EQ(VM::max_memory_size(1), evmone::Memory::page_size);
    EXPECT_EQ(VM::max_memory_size(int64_t{1} << 25), VM::MAX_RESERVED_MEMORY);
    EXPECT_EQ(VM::max_memory_size(std::numeric_limits<int64_t>::max()), VM::MAX_RESERVED_MEMORY);

    for (int64_t gas = 1; gas < (int64_t{1} << 25); gas += gas / 64 + 1)
    {
        const auto size = VM::max_memory_size(gas);
        ASSERT_EQ(size % 32, 0) << gas;
        ASSERT_LE(size, VM::MAX_RESERVED_MEMORY) << gas;
        ASSERT_GE(size, evmone::Memory::page_size) << gas;

        // The largest affordable memory fits, unless capped.
        const auto words = size / 32;
        if (size < VM::MAX_RESERVED_MEMORY)
            ASSERT_GT(cost(words + 1), static_cast<uint64_t>(gas)) << gas;

        // Not much more than needed once the w²/512 term dominates: 16 * sqrt(2 * gas) words,
        // overestimated by the sqrt bound, stay within 50% of the largest affordable size.
        if (gas >= 1'000'000)
        {
            uint64_t max_words = 0;
            for (uint64_t step = uint64_t{1} << 20; step != 0; step /= 2)
            {
                if (cost(max_words + step) <= static_cast<uint64_t>(gas))
                    max_words += step;
            }
            ASSERT_LE(words, max_words * 3 / 2) << gas;
        }
    }

    // The transaction gas limit cap (EIP-7825) fits.
    EXPECT_GE(VM::max_memory_size(int64_t{1} << 24), 91'917 * 32);
    EXPECT_LT(VM::max_memory_size(int64_t{1} << 24), VM::MAX_RESERVED_MEMORY);
}
