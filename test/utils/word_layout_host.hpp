// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/mocked_host.hpp>
#include <evmone/word_layout.hpp>

#ifdef EVMONE_WORD_LAYOUT
namespace evmone::test
{
/// MockedHost keeping at its boundary to the contract of the word layout
/// (evmone/word_layout.hpp) as the state Host does: the calldata of a call and the data of a log
/// come in the layout and are recorded in byte order, without the layout's message flags, and the
/// output of a call goes back in the layout, in storage from wl::alloc_output().
class WordLayoutMockedHost : public evmc::MockedHost
{
public:
    evmc::Result call(const evmc_message& msg) noexcept override
    {
        auto byte_msg = msg;
        byte_msg.flags &= ~(wl::FLAG_WORD_INPUT | wl::FLAG_WORD_OUTPUT);
        evmc::bytes input;
        if ((msg.flags & wl::FLAG_WORD_INPUT) != 0 && msg.input_size != 0)
        {
            input = wl::to_bytes(msg.input_data, msg.input_size);
            byte_msg.input_data = input.data();
        }
        auto result = MockedHost::call(byte_msg);
        if ((msg.flags & wl::FLAG_WORD_OUTPUT) == 0 || result.output_size == 0)
            return result;

        // A new result, so that the test's own release function sees its output.
        auto word_result = result.raw();
        auto* const output = wl::alloc_output(word_result.output_size);
        wl::copy_b2w(output, word_result.output_data, word_result.output_size);
        word_result.output_data = output;
        word_result.release = wl::free_output;
        return evmc::Result{word_result};
    }

    void emit_log(const evmc::address& addr, const uint8_t* data, size_t data_size,
        const evmc::bytes32 topics[], size_t topics_count) noexcept override
    {
        const auto byte_data = wl::to_bytes(data, data_size);
        MockedHost::emit_log(addr, byte_data.data(), data_size, topics, topics_count);
    }
};
}  // namespace evmone::test
#endif
