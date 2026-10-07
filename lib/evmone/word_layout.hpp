// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

/// EVMONE_WORD_LAYOUT: EVM memory is kept in the big-endian word layout on rv32, where the guest
/// has no byte-reversing instruction and MLOAD, MSTORE and CALLDATALOAD would reverse 32 bytes
/// every time. The layout is a mapping of addresses: the byte at logical address a is stored at
/// physical address a ^ 3, so each aligned word holds the big-endian number of its 4 bytes and an
/// aligned 32-byte load is 8 word copies in reversed order. Every pointer into a buffer in this
/// layout ("W pointer") is a logical address, and alignment is always that of the address, never
/// of the offset in the EVM memory: the calldata of a callee is a W pointer into the caller's
/// memory at any alignment.
///
/// Memory, the calldata of a call, the return data and the output of a frame at depth > 0 are in
/// this layout. Everything else (the calldata of a transaction, code, the outputs of a
/// transaction, logs, hash inputs, the inputs of precompiles) is byte order, and each boundary
/// converts with one copy. The test build turns the layout on in every C++ file with
/// EVMONE_RV32_DISPATCH_TEST. It needs a host that follows the contract above, i.e. the state
/// Host; the Host of evmone's own unit tests (built without the layout) does not.
#if (defined(AIRBENDER) && defined(__riscv) && __riscv_xlen == 32) || \
    defined(EVMONE_RV32_DISPATCH_TEST)
#define EVMONE_WORD_LAYOUT 1
#endif

#ifdef EVMONE_WORD_LAYOUT
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

namespace evmone::wl
{
/// Message flags of an evmc_message built by an EVM frame (see call_impl). The input is a W
/// pointer into the caller's memory; the output is wanted as W data, phase 0, with its storage
/// rounded up to a multiple of 4 bytes (see make_execution_result). Messages of a transaction or a
/// system call carry neither: their input is byte order and so is their output.
inline constexpr uint32_t FLAG_WORD_INPUT = 0x40000000;
inline constexpr uint32_t FLAG_WORD_OUTPUT = 0x80000000;

/// Words are accessed only through this named type: GCC drops the stores of one reached by auto.
using word_t = uint32_t __attribute__((may_alias));

[[nodiscard]] inline constexpr size_t round_up4(size_t n) noexcept { return (n + 3) & ~size_t{3}; }

#ifdef EVMONE_RV32_DISPATCH_TEST
/// What the conversions cost, counted in the native test build only: its tests bound it by the
/// gas a program pays.
struct Usage
{
    uint64_t converted = 0;  ///< Bytes copied from W data into byte order.
};
inline Usage usage;
#endif

/// The byte at logical address p + i.
[[nodiscard]] inline uint8_t* at(const uint8_t* p, size_t i) noexcept
{
    return reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(p + i) ^ 3);
}

[[nodiscard]] inline bool is_aligned4(const void* p) noexcept
{
    return (reinterpret_cast<uintptr_t>(p) & 3) == 0;
}

namespace detail
{
/// The offset from the W pointer p to the byte that holds its logical byte i, when p has the
/// alignment @p ph (bytes past a multiple of 4). It can be negative: the word starts before p.
[[nodiscard]] constexpr int phys_offset(unsigned ph, unsigned i) noexcept
{
    const unsigned q = ph + i;
    return static_cast<int>((q & ~3u) + ((q & 3u) ^ 3u)) - static_cast<int>(ph);
}

/// The 32 bytes at the W pointer p, reversed into the little-endian uint256 at x.
template <unsigned Ph>
inline void load_phase(uint8_t* x, const uint8_t* p) noexcept
{
#pragma GCC unroll 32
    for (unsigned i = 0; i < 32; ++i)
        x[31 - i] = p[phys_offset(Ph, i)];
}

/// The big-endian bytes of the uint256 at x, stored at the W pointer p.
template <unsigned Ph>
inline void store_phase(uint8_t* p, const uint8_t* x) noexcept
{
#pragma GCC unroll 32
    for (unsigned i = 0; i < 32; ++i)
        p[phys_offset(Ph, i)] = x[31 - i];
}
}  // namespace detail

/// load_u256() for an address that is not word aligned. Out of line, so that the aligned case in
/// the interpreter keeps the registers it has.
[[gnu::noinline]] inline void load_u256_unaligned(intx::uint256& x, const uint8_t* p) noexcept
{
    auto* const xb = reinterpret_cast<uint8_t*>(&x);
    switch (reinterpret_cast<uintptr_t>(p) & 3)
    {
    case 1:
        return detail::load_phase<1>(xb, p);
    case 2:
        return detail::load_phase<2>(xb, p);
    default:
        return detail::load_phase<3>(xb, p);
    }
}

/// store_u256() for an address that is not word aligned.
[[gnu::noinline]] inline void store_u256_unaligned(uint8_t* p, const intx::uint256& x) noexcept
{
    const auto* const xb = reinterpret_cast<const uint8_t*>(&x);
    switch (reinterpret_cast<uintptr_t>(p) & 3)
    {
    case 1:
        return detail::store_phase<1>(p, xb);
    case 2:
        return detail::store_phase<2>(p, xb);
    default:
        return detail::store_phase<3>(p, xb);
    }
}

/// Loads the 32 bytes at the W pointer p as a big-endian number into x, which must not overlap
/// them. A word-aligned p holds the number's words already, most significant first: 8 word
/// copies, as one asm block with one scratch register (as separate C copies GCC loads all 8 words
/// first and spills around them in the interpreter loop).
[[gnu::always_inline]] inline void load_u256(intx::uint256& x, const uint8_t* p) noexcept
{
    if (is_aligned4(p)) [[likely]]
    {
#if defined(__riscv) && __riscv_xlen == 32
        using Bytes = uint8_t[32];
        uint32_t t;
#define EVMONE_WL(si, di) "lw %[t], " #si "(%[s])\n\tsw %[t], " #di "(%[d])\n\t"
        asm(EVMONE_WL(0, 28) EVMONE_WL(4, 24) EVMONE_WL(8, 20) EVMONE_WL(12, 16) EVMONE_WL(16, 12)
                EVMONE_WL(20, 8) EVMONE_WL(24, 4) EVMONE_WL(28, 0)
            : [t] "=&r"(t), "=m"(*reinterpret_cast<Bytes*>(&x))
            : [d] "r"(&x), [s] "r"(p), "m"(*reinterpret_cast<const Bytes*>(p)));
#undef EVMONE_WL
#else
        auto* const d = reinterpret_cast<word_t*>(&x);
        const auto* const s = reinterpret_cast<const word_t*>(p);
        for (unsigned k = 0; k < 8; ++k)
            d[7 - k] = s[k];
#endif
    }
    else
        load_u256_unaligned(x, p);
}

/// Stores x as 32 big-endian bytes at the W pointer p, see load_u256().
[[gnu::always_inline]] inline void store_u256(uint8_t* p, const intx::uint256& x) noexcept
{
    if (is_aligned4(p)) [[likely]]
    {
#if defined(__riscv) && __riscv_xlen == 32
        using Bytes = uint8_t[32];
        uint32_t t;
#define EVMONE_WL(si, di) "lw %[t], " #si "(%[s])\n\tsw %[t], " #di "(%[d])\n\t"
        asm(EVMONE_WL(28, 0) EVMONE_WL(24, 4) EVMONE_WL(20, 8) EVMONE_WL(16, 12) EVMONE_WL(12, 16)
                EVMONE_WL(8, 20) EVMONE_WL(4, 24) EVMONE_WL(0, 28)
            : [t] "=&r"(t), "=m"(*reinterpret_cast<Bytes*>(p))
            : [d] "r"(p), [s] "r"(&x), "m"(*reinterpret_cast<const Bytes*>(&x)));
#undef EVMONE_WL
#else
        auto* const d = reinterpret_cast<word_t*>(p);
        const auto* const s = reinterpret_cast<const word_t*>(&x);
        for (unsigned k = 0; k < 8; ++k)
            d[k] = s[7 - k];
#endif
    }
    else
        store_u256_unaligned(p, x);
}

/// Loads the first @p len < 32 bytes at the W pointer p, followed by zeros, as a big-endian
/// number. The bytes after them in the same word belong to someone else and are not read.
[[gnu::noinline]] inline void load_u256_partial(
    intx::uint256& x, const uint8_t* p, size_t len) noexcept
{
    uint8_t b[32] = {};
    for (size_t i = 0; i < len; ++i)
        b[i] = *at(p, i);
    x = intx::be::load<intx::uint256>(b);
}

/// Copies @p n bytes from the byte-order buffer s to the W pointer d. Only [d, d + n) is written.
inline void copy_b2w(uint8_t* d, const uint8_t* s, size_t n) noexcept
{
    size_t i = 0;
    for (; i < n && !is_aligned4(d + i); ++i)
        *at(d, i) = s[i];
    // An aligned word of d holds the big-endian number of the 4 bytes: 4 byte copies, which cost
    // what a byte reversal in registers does but need no alignment of s.
    for (; i + 4 <= n; i += 4)
    {
        uint8_t* const w = d + i;
        w[3] = s[i];
        w[2] = s[i + 1];
        w[1] = s[i + 2];
        w[0] = s[i + 3];
    }
    for (; i < n; ++i)
        *at(d, i) = s[i];
}

/// Copies @p n bytes from the W pointer s to the byte-order buffer d.
inline void copy_w2b(uint8_t* d, const uint8_t* s, size_t n) noexcept
{
#ifdef EVMONE_RV32_DISPATCH_TEST
    usage.converted += n;
#endif
    size_t i = 0;
    for (; i < n && !is_aligned4(s + i); ++i)
        d[i] = *at(s, i);
    for (; i + 4 <= n; i += 4)
    {
        const uint8_t* const w = s + i;
        d[i] = w[3];
        d[i + 1] = w[2];
        d[i + 2] = w[1];
        d[i + 3] = w[0];
    }
    for (; i < n; ++i)
        d[i] = *at(s, i);
}

/// Copies @p n bytes between W pointers. Only [d, d + n) is written. With @p Overlap the ranges may
/// overlap, as memmove() allows, and the middle words move with it; without, memcpy() does.
template <bool Overlap>
inline void copy_w2w_impl(uint8_t* d, const uint8_t* s, size_t n) noexcept
{
    if (n == 0 || d == s)
        return;
    if (((reinterpret_cast<uintptr_t>(d) ^ reinterpret_cast<uintptr_t>(s)) & 3) == 0)
    {
        // Congruent: the whole words in the middle are the same bytes in both buffers, so they
        // move as memory. The 0 to 3 bytes at each end are read before the move and written
        // after it: a word move would take the neighbours' bytes along, and the move may
        // overwrite them.
        const size_t head = std::min(n, (4 - (reinterpret_cast<uintptr_t>(d) & 3)) & 3);
        const size_t words = (n - head) / 4;
        const size_t tail_at = head + words * 4;
        uint8_t edge[6];
        for (size_t k = 0; k < head; ++k)
            edge[k] = *at(s, k);
        for (size_t k = tail_at; k < n; ++k)
            edge[3 + k - tail_at] = *at(s, k);
        if (words != 0)
        {
            if constexpr (Overlap)
                std::memmove(d + head, s + head, words * 4);
            else
                std::memcpy(d + head, s + head, words * 4);
        }
        for (size_t k = 0; k < head; ++k)
            *at(d, k) = edge[k];
        for (size_t k = tail_at; k < n; ++k)
            *at(d, k) = edge[3 + k - tail_at];
    }
    else if (d < s)
    {
        for (size_t i = 0; i < n; ++i)
            *at(d, i) = *at(s, i);
    }
    else
    {
        for (size_t i = n; i-- != 0;)
            *at(d, i) = *at(s, i);
    }
}

/// Copies between W buffers that do not overlap.
inline void copy_w2w(uint8_t* d, const uint8_t* s, size_t n) noexcept
{
    copy_w2w_impl<false>(d, s, n);
}

/// Copies within W memory (MCOPY), where the ranges may overlap.
inline void move_w2w(uint8_t* d, const uint8_t* s, size_t n) noexcept
{
    copy_w2w_impl<true>(d, s, n);
}

/// Zeroes @p n bytes at the W pointer d. Only [d, d + n) is written.
inline void zero(uint8_t* d, size_t n) noexcept
{
    size_t i = 0;
    for (; i < n && !is_aligned4(d + i); ++i)
        *at(d, i) = 0;
    if (const size_t words = (n - i) / 4; words != 0)
    {
        std::memset(d + i, 0, words * 4);
        i += words * 4;
    }
    for (; i < n; ++i)
        *at(d, i) = 0;
}

/// Reverses the bytes of every 32-bit word of the first @p words words at p: byte order data of
/// the words' bytes to W data, phase 0, and back. Byte copies take two instructions a byte, where
/// __builtin_bswap32 is a call into libgcc on rv32im; the barrier keeps GCC from merging them.
inline void swap_words(uint8_t* p, size_t words) noexcept
{
    for (size_t k = 0; k < words; ++k, p += 4)
    {
        uint8_t b0 = p[0], b1 = p[1];
        asm("" : "+r"(b0), "+r"(b1));
        p[0] = p[3];
        p[1] = p[2];
        p[2] = b1;
        p[3] = b0;
    }
}

/// The n bytes at the W pointer p as a byte string, written once: a string constructed to a size
/// first zeroes it.
[[nodiscard]] inline evmc::bytes to_bytes(const uint8_t* p, size_t n)
{
    evmc::bytes b;
    const auto fill = [p](uint8_t* buffer, size_t size) noexcept {
        copy_w2b(buffer, p, size);
        return size;
    };
#ifdef __cpp_lib_string_resize_and_overwrite
    b.resize_and_overwrite(n, fill);
#else
    b.__resize_and_overwrite(n, fill);  // libstdc++'s name for it before C++23
#endif
    return b;
}

/// A byte buffer that only grows. The guest never frees memory, so a buffer that is made again for
/// every use of the same kind (the input of a precompile or the data of a hash over 50000 bytes,
/// 50000 times in a block) would exhaust it: these keep and reuse their storage.
class Buffer
{
    std::unique_ptr<uint8_t[]> m_data;
    size_t m_capacity = 0;

public:
    /// The storage, of at least @p n bytes, 8-byte aligned. Its previous contents are lost.
    [[nodiscard]] uint8_t* get(size_t n) noexcept
    {
        if (n > m_capacity)
        {
            m_data.reset(new uint8_t[n]);
            m_capacity = n;
        }
        return m_data.get();
    }

    [[nodiscard]] const uint8_t* data() const noexcept { return m_data.get(); }
};

/// A buffer of @p n bytes that the hash, the precompiles and the code copy share: they use it only
/// within one call that does not run the EVM, so no use is open when the next starts. A larger
/// request keeps the contents of a smaller one in the same call. It starts
/// @p Offset bytes after an 8-byte boundary. The hash wants the boundary (it reads 64-bit lanes);
/// the precompiles take an offset of 4, which is no multiple of 32: the copies they make from
/// their input then take the word path, not the BigInt MEMCOPY that a source aligned to 32 bytes
/// would.
template <size_t Offset = 0>
[[nodiscard]] inline uint8_t* scratch(size_t n) noexcept
{
    static_assert(Offset % 4 == 0 && Offset < 8);
    // A pointer and a size, not a Buffer: no destructor to register. One thread and no
    // thread-local storage on the guest.
#ifdef AIRBENDER
    static uint8_t* storage = nullptr;
    static size_t capacity = 0;
#else
    thread_local uint8_t* storage = nullptr;
    thread_local size_t capacity = 0;
#endif
    if (n + Offset > capacity)
    {
        // The contents are kept: MODEXP's input grows once its first bytes are in.
        auto* const grown = new uint8_t[n + Offset];
        std::copy_n(storage, capacity, grown);
        delete[] storage;
        storage = grown;
        capacity = n + Offset;
    }
    return storage + Offset;
}

/// The byte-order copy of the @p n bytes at the W pointer p, in scratch().
template <size_t Offset = 0>
[[nodiscard]] inline const uint8_t* scratch_copy(const uint8_t* p, size_t n) noexcept
{
    uint8_t* const copy = scratch<Offset>(n);
    copy_w2b(copy, p, n);
    return copy;
}

/// The return data of a frame: the output of the last call, W data of phase 0 whose storage is
/// rounded up to whole words, with the size in bytes kept apart. A byte string would drop the
/// partial last word and put its terminator on a byte of the data.
class ReturnData
{
    Buffer m_buffer;
    size_t m_size = 0;

public:
    [[nodiscard]] size_t size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] const uint8_t* data() const noexcept { return m_buffer.data(); }
    void clear() noexcept { m_size = 0; }

    /// Takes the W data of phase 0 and @p size bytes at @p src, whose storage is rounded up.
    void assign(const uint8_t* src, size_t size) noexcept
    {
        if (size != 0)
            std::memcpy(m_buffer.get(round_up4(size)), src, round_up4(size));
        m_size = size;
    }
};
}  // namespace evmone::wl
#endif
