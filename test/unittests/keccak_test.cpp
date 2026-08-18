// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0

/// Tests for the keccak absorb, which under KECCAK_STRICT_ALIGNMENT reads an unaligned input as
/// aligned words and so reaches outside it in both directions. Both builds of the absorb are
/// covered: the one the host target selects and the strict-alignment one (see CMakeLists).

#include <evmc/hex.hpp>
#include <evmone_precompiles/keccak.h>
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define EVMONE_HAVE_GUARD_PAGES 1
#endif

extern "C" {
ethash_hash256 ethash_keccak256_strict(const uint8_t* data, size_t size) noexcept;
ethash_hash256 ethash_keccak256_32_strict(const uint8_t data[32]) noexcept;
}

/// One build of the absorb under test. Not in an anonymous namespace: it is the parameter type of
/// the test classes TEST_P declares below, which have external linkage.
struct Impl
{
    const char* name;
    ethash_hash256 (*hash)(const uint8_t*, size_t) noexcept;
    ethash_hash256 (*hash32)(const uint8_t[32]) noexcept;
};

class keccak : public testing::TestWithParam<Impl>
{};

namespace
{
constexpr size_t WORD_SIZE = sizeof(uint64_t);

std::string to_hex(const ethash_hash256& h)
{
    return evmc::hex({h.bytes, std::size(h.bytes)});
}

INSTANTIATE_TEST_SUITE_P(absorb, keccak,
    testing::Values(Impl{"native", ethash_keccak256, ethash_keccak256_32},
        Impl{"strict_alignment", ethash_keccak256_strict, ethash_keccak256_32_strict}),
    [](const testing::TestParamInfo<Impl>& i) { return i.param.name; });
}  // namespace

TEST_P(keccak, known_vectors)
{
    const auto& impl = GetParam();
    EXPECT_EQ(to_hex(impl.hash(nullptr, 0)),
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");

    const std::string abc = "abc";
    EXPECT_EQ(to_hex(impl.hash(reinterpret_cast<const uint8_t*>(abc.data()), abc.size())),
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
}

TEST_P(keccak, independent_of_input_alignment)
{
    // The same bytes at every alignment must hash the same. This is what the word reader's carry
    // and shifts have to get right, and checking it needs no reference implementation.
    const auto& impl = GetParam();
    std::vector<uint8_t> content(600);
    for (size_t i = 0; i < content.size(); ++i)
        content[i] = static_cast<uint8_t>(i * 31 + 7);

    alignas(WORD_SIZE) std::array<uint8_t, 600 + WORD_SIZE> buffer{};
    for (size_t size = 0; size <= content.size(); ++size)
    {
        const auto expected = to_hex(impl.hash(content.data(), size));
        for (size_t phase = 0; phase < WORD_SIZE; ++phase)
        {
            std::memcpy(&buffer[phase], content.data(), size);
            ASSERT_EQ(to_hex(impl.hash(&buffer[phase], size)), expected)
                << "size " << size << " phase " << phase;
        }
    }
}

TEST_P(keccak, keccak256_32_matches_the_general_entry_point)
{
    const auto& impl = GetParam();
    alignas(WORD_SIZE) std::array<uint8_t, 32 + WORD_SIZE> buffer{};
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] = static_cast<uint8_t>(i * 13 + 1);

    for (size_t phase = 0; phase < WORD_SIZE; ++phase)
        EXPECT_EQ(to_hex(impl.hash32(&buffer[phase])), to_hex(impl.hash(&buffer[phase], 32)))
            << "phase " << phase;
}

#ifdef EVMONE_HAVE_GUARD_PAGES
TEST_P(keccak, reads_no_page_outside_the_input)
{
    // The words holding the input's first and last bytes may reach outside it, which is safe only
    // because an aligned word cannot span two pages. Run in a page fenced on both sides, so a read
    // that leaves the containing word in either direction raises SIGSEGV.
    const auto& impl = GetParam();
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* const p = static_cast<uint8_t*>(
        mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(p, MAP_FAILED);
    ASSERT_EQ(mprotect(p, page, PROT_NONE), 0);              // Before the input.
    ASSERT_EQ(mprotect(p + 2 * page, page, PROT_NONE), 0);   // After it.
    std::memset(p + page, 0xab, page);

    // The checks are that these do not fault.
    for (size_t size = 1; size <= 300; ++size)
        impl.hash(p + 2 * page - size, size);  // Last byte flush against the upper guard.
    for (size_t phase = 0; phase < WORD_SIZE; ++phase)
        for (size_t size = 1; size <= 300; ++size)
            impl.hash(p + page + phase, size);  // First byte against the lower guard.

    EXPECT_EQ(munmap(p, 3 * page), 0);
}
#endif
