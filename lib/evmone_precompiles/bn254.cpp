// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2026 The zvm1 Authors (modifications)
// Copyright 2023 The evmone Authors (original)
// SPDX-License-Identifier: Apache-2.0

#include "bn254.hpp"
#include <algorithm>
#include <new>

#if (defined(AIRBENDER) && defined(__riscv)) || defined(EVMONE_RV32_DISPATCH_TEST)
/// mul() by width-5 NAFs of the GLV halves over a table of odd multiples, with the point
/// operations in place (see mul_wnaf()). EVMONE_RV32_DISPATCH_TEST builds it on the host for
/// testing.
#define EVMONE_BN254_MUL_WNAF 1
#else
#define EVMONE_BN254_MUL_WNAF 0
#endif

namespace evmone::crypto::bn254
{
static_assert(AffinePoint{} == 0, "default constructed is the point at infinity");

bool validate(const AffinePoint& pt) noexcept
{
    const auto yy = pt.y * pt.y;
    const auto xxx = pt.x * pt.x * pt.x;
    const auto on_curve = yy == xxx + Curve::B;
    return on_curve || pt == 0;
}

#if EVMONE_BN254_MUL_WNAF
namespace
{
using Point = ecc::ProjPoint<Curve>;

/// The signed digit width of both halves, and the table of odd multiples 1P, 3P, ..., 15P.
constexpr unsigned WNAF_W = 5;
constexpr size_t TABLE_SIZE = size_t{1} << (WNAF_W - 2);
/// Digits of a width-5 NAF of a half below 2^128: the carry may add one more.
constexpr unsigned WNAF_LEN = 129;
// ecc::decompose() rounds both lattice coefficients to the nearest integer, which bounds |k1| by
// (|x1| + |x2|) / 2 and |k2| by (|y1| + |y2|) / 2: both below 2^128, all that wnaf() reads.
static_assert((Curve::X1 + Curve::X2) / 2 < (uint256{1} << 128) &&
              (Curve::MINUS_Y1 + Curve::Y2) / 2 < (uint256{1} << 128));

/// 1 in Montgomery form, folded at compile time (Fq::one() at run time is a CSR multiplication).
constexpr const auto& FP_ONE = Point::ONE;

/// v, which on rv32 the compiler is told is 32-byte aligned, so that the ModArith operations
/// inlined into dbl_inplace() and madd_inplace() fold their alignment tests (as in
/// secp256k1.cpp). Every caller passes such storage: FieldElement's value_ is alignas(32) there.
template <typename T>
[[gnu::always_inline]] inline T& aligned(T& v) noexcept
{
#if defined(AIRBENDER) && defined(__riscv)
    static_assert(alignof(T) == 32);
    return *static_cast<T*>(__builtin_assume_aligned(&v, 32));
#else
    return v;
#endif
}

/// Writes the width-5 NAF of k < 2^128: naf[i] is the digit of 2^i, 0 or odd with |naf[i]| < 16,
/// and at least 4 zeros follow each non-zero one. naf must be zeroed (WNAF_LEN digits). Returns
/// the index past the top non-zero digit. This is secp256k1.cpp's wnaf() (libsecp256k1's
/// secp256k1_ecmult_wnaf()), with k's 32-bit words taken out of its 64-bit ones.
unsigned wnaf(int8_t* naf, const uint256& k) noexcept
{
    const uint32_t x[6] = {static_cast<uint32_t>(k[0]), static_cast<uint32_t>(k[0] >> 32),
        static_cast<uint32_t>(k[1]), static_cast<uint32_t>(k[1] >> 32), 0, 0};
    // Trailing zeros of a non-zero word: rv32im has no ctz, so a de Bruijn lookup (5 instructions).
    static constexpr uint8_t DEBRUIJN[32] = {0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17,
        4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};
    const auto ctz = [](uint32_t v) noexcept {
        return static_cast<unsigned>(DEBRUIJN[((v & (0u - v)) * 0x077CB531u) >> 27]);
    };
    // The `count` bits at `pos`, from at most two adjacent words.
    const auto get_bits = [&x](unsigned pos, unsigned count) noexcept {
        const unsigned wi = pos / 32;
        const unsigned sh = pos % 32;
        uint32_t v = x[wi] >> sh;
        if (sh != 0)
            v |= x[wi + 1] << (32 - sh);
        return v & ((uint32_t{1} << count) - 1);
    };
    unsigned bit = 0;
    unsigned len = 0;
    uint32_t carry = 0;
    while (bit < WNAF_LEN)
    {
        // The next bit that differs from the carry, found a word at a time.
        const unsigned wi = bit / 32;
        const uint32_t differing = (x[wi] ^ (0u - carry)) >> (bit % 32);
        if (differing == 0)
        {
            bit = (wi + 1) * 32;
            continue;
        }
        bit += ctz(differing);
        if (bit >= WNAF_LEN)
            break;
        const unsigned now = std::min(WNAF_W, WNAF_LEN - bit);
        const auto word = get_bits(bit, now) + carry;
        carry = (word >> (WNAF_W - 1)) & 1;
        naf[bit] = static_cast<int8_t>(static_cast<int>(word) - static_cast<int>(carry << WNAF_W));
        len = bit + 1;
        bit += now;
    }
    return len;
}

/// p = 2p in place, as the representative (X3/4 : Y3/8 : Z3/2) of ecc::dbl()'s a = 0 result:
/// with L = 3X^2/2 and T = XY^2, X' = L^2 - 2T, Y' = L(T - X') - Y^4 and Z' = YZ. As
/// secp256k1.cpp's dbl_inplace(): the point is used only projectively, so one halving of X
/// replaces the doublings in 2YZ, 4XY^2 and 8Y^4, and each coordinate is written into p as it
/// dies. The point at infinity (z == 0) stays at infinity.
__attribute__((flatten)) void dbl_inplace(Point& p_) noexcept
{
    auto& p = aligned(p_);
    auto& [x1, y1, z1] = p;
    DECL_FE_COPY(Fq, yy, y1); yy *= y1;          // Y^2
    z1 *= y1;                                    // Z' = YZ
    DECL_FE_COPY(Fq, m, x1); m.halve();          // X/2
    m += x1; m *= x1;                            // L = 3X^2/2
    y1 = yy; y1 *= yy;                           // Y^4
    yy *= x1;                                    // T = XY^2
    x1 = m; x1 *= m;                             // L^2
    x1 -= yy; x1 -= yy;                          // X' = L^2 - 2T
    yy -= x1; yy *= m;                           // L(T - X')
    y1.rsub(yy);                                 // Y' = L(T - X') - Y^4
}

/// a == 0, testing the low word first: h is almost never 0, so the common case is a load and a
/// branch where the full test ORs all 8 words (as secp256k1.cpp's is_zero_low_first()).
[[gnu::always_inline]] inline bool is_zero_low_first(const Fq& a) noexcept
{
    typedef uint32_t __attribute__((may_alias)) word;
    if (*reinterpret_cast<const word*>(&a) != 0) [[likely]]
        return false;
    asm volatile("");  // keeps GCC from merging the low-word test back into the 8-word OR below
    return a == 0;
}

/// p += (x2, y2), an affine point other than infinity, with p not infinity either, in place: the
/// add-1998-cmo-2 formula with z2 = 1, written into p's own coordinates as each one dies (as
/// secp256k1.cpp's madd_inplace<true>()). The new z is the old one times h = x2 z1^2 - x1; with
/// Ratio, h is also constructed in *zr (0 when p == (x2, y2), which doubles p instead: Ratio
/// callers must exclude that case, as odd_multiples() does). Returns true if the sum is the point
/// at infinity (p == -(x2, y2)), and then leaves p with z == 0.
template <bool Ratio = false>
__attribute__((flatten)) bool madd_inplace(
    Point& p_, const Fq& x2_, const Fq& y2_, [[maybe_unused]] Fq* zr = nullptr) noexcept
{
    auto& p = aligned(p_);
    const auto& x2 = aligned(x2_);
    const auto& y2 = aligned(y2_);
    auto& [x1, y1, z1] = p;
    DECL_FE_COPY(Fq, z1z1, z1); z1z1 *= z1;      // z1^2
    DECL_FE_COPY(Fq, h, x2); h *= z1z1;          // u2 = x2 z1^2
    z1z1 *= z1; z1z1 *= y2;                      // s2 = y2 z1^3
    h -= x1;                                     // h = u2 - x1
    z1z1 -= y1;                                  // r = s2 - y1
    if constexpr (Ratio)
        new (zr) Fq{h};
    if (is_zero_low_first(h)) [[unlikely]]
    {
        if (z1z1 == 0)  // p == (x2, y2)
        {
            dbl_inplace(p);
            return false;
        }
        z1 = Fq{};  // p == -(x2, y2): the sum is the point at infinity.
        return true;
    }
    z1 *= h;                                     // z3 = z1 h
    DECL_FE_COPY(Fq, hh, h); hh *= h;            // h^2
    h *= hh;                                     // h^3
    hh *= x1;                                    // v = x1 h^2
    x1 = z1z1; x1 *= z1z1;                       // r^2
    x1 -= h; x1 -= hh; x1 -= hh;                 // x3 = r^2 - h^3 - 2v
    hh -= x1;                                    // v - x3
    y1 *= h;                                     // y1 h^3
    z1z1 *= hh;                                  // r (v - x3)
    y1.rsub(z1z1);                               // y3 = r (v - x3) - y1 h^3
    return false;
}

/// The odd multiples (2j+1)P, j < 8, of P != O on a common z, with libsecp256k1's
/// isomorphic-curve technique (secp256k1_ecmult_odd_multiples_table()): (x, y) -> (u^2 x, u^3 y)
/// maps the curve onto y^2 = x^3 + u^6 b, and the a = 0 doubling and addition formulas do not
/// involve b, so a sum can be computed on the image and mapped back by multiplying its z by u.
/// 2P = (X : Y : u) is the affine point (X, Y) on the image for u, where (2j+1)P = (2j-1)P + 2P
/// are mixed additions, each multiplying z by the ratio h it returns. Scaling every entry to the
/// z of the last one by the product of the later ratios puts them all on that z, z_7. Returns
/// Z = u z_7: on the image for Z, entry j is the affine point (x[j], y[j]). No addition here meets
/// P == +/-Q, which needs (2j+1)P = O or (2j-3)P = O in a group of prime order. Constructs
/// x[0..7] and y[0..7]. Out of line: inlined, it grows mul_wnaf() past GCC's inlining limits, and
/// the guest then calls mul_wnaf() and madd_inplace<true>() instead of inlining them, 110 cycles
/// more per multiplication.
[[gnu::noinline]] Fq odd_multiples(const AffinePoint& pt, Fq* x, Fq* y) noexcept
{
    alignas(32) std::byte zr_raw[TABLE_SIZE * sizeof(Fq)];
    auto* const zr = reinterpret_cast<Fq*>(zr_raw);  // z_j / z_(j-1)

    // 2P = (X : Y : u), and P on the image for u: (u^2 x, u^3 y).
    Point two_p{pt.x, pt.y, FP_ONE};
    dbl_inplace(two_p);
    DECL_FE_COPY(Fq, uu, two_p.z); uu *= two_p.z;
    new (&x[0]) Fq{pt.x};
    x[0] *= uu;
    uu *= two_p.z;
    new (&y[0]) Fq{pt.y};
    y[0] *= uu;

    Point acc{x[0], y[0], FP_ONE};
    for (size_t j = 1; j < TABLE_SIZE; ++j)
    {
        madd_inplace<true>(acc, two_p.x, two_p.y, &zr[j]);
        new (&x[j]) Fq{acc.x};
        new (&y[j]) Fq{acc.y};
    }
    // Entry j has z_j = zr[1]...zr[j]: bring it to z_7 with s = zr[j+1]...zr[7], x s^2 and y s^3.
    DECL_FE_COPY(Fq, s, zr[TABLE_SIZE - 1]);
    for (size_t j = TABLE_SIZE - 1; j-- != 0;)
    {
        DECL_FE_COPY(Fq, ss, s); ss *= s;
        x[j] *= ss;
        ss *= s;
        y[j] *= ss;
        if (j != 0)
            s *= zr[j];
    }
    acc.z *= two_p.z;  // u z_7
    return acc.z;
}

/// sum += (xs[j], ys[d < 0][j]) for the non-zero digit d = +/-(2j + 1). The first addition
/// constructs the sum and the others know it is a live point, until one cancels it: then the next
/// addition constructs it again (see secp256k1.cpp's msm_wnaf()).
[[gnu::always_inline]] inline void add_entry(
    Point& sum, bool& started, int d, const Fq* xs, const Fq* const ys[2]) noexcept
{
    const auto j = static_cast<size_t>((d > 0 ? d : -d) >> 1);
    const auto& x = xs[j];
    const auto& y = ys[d < 0][j];
    if (started)
        started = !madd_inplace(sum, x, y);
    else
    {
        new (&sum) Point{x, y, FP_ONE};
        started = true;
    }
}

/// [c]P as [k1]P + [k2]phi(P) with ecc::decompose()'s halves, phi(x, y) = (BETA x, y) being
/// [LAMBDA]P, from width-5 NAFs of |k1| and |k2| over the odd multiples of odd_multiples() and
/// their phi images, a half's sign folded into its digits. The sum is computed on the image curve
/// where the table entries are affine, with mixed additions instead of Jacobian ones (phi maps
/// that curve to itself), and mapped back by the table's z.
///
/// A main loop addition can meet P == -Q: c = r, 3r or 5r decompose into a non-zero (k1, k2) on
/// the lattice whose last addition cancels the sum. Such an addition restarts the sum, as in
/// secp256k1.cpp's msm_wnaf(); P == Q, which no input reaches, doubles (madd_inplace()).
Point mul_wnaf(const AffinePoint& pt, const uint256& c) noexcept
{
    const auto [k1, k2] = ecc::decompose<Curve>(c);

    alignas(4) int8_t naf_a[WNAF_LEN + 3]{};
    alignas(4) int8_t naf_b[WNAF_LEN + 3]{};
    const auto top = std::max(wnaf(naf_a, k1.value), wnaf(naf_b, k2.value));
    if (top == 0)  // c is a multiple of the group order.
        return {};

    alignas(32) std::byte t_raw[4 * TABLE_SIZE * sizeof(Fq)];
    auto* const xs = reinterpret_cast<Fq*>(t_raw);  // x_j
    auto* const bx = xs + TABLE_SIZE;               // BETA x_j
    auto* const ys = bx + TABLE_SIZE;               // y_j
    auto* const ny = ys + TABLE_SIZE;               // -y_j
    const auto z = odd_multiples(pt, xs, ys);
    for (size_t j = 0; j < TABLE_SIZE; ++j)
    {
        new (&bx[j]) Fq{xs[j]};
        bx[j] *= Curve::BETA;
        new (&ny[j]) Fq{};
        ny[j] -= ys[j];
    }

    // A digit d selects x from xs (k1's half) or bx (k2's half), and y or -y by its sign and the
    // half's.
    const Fq* const ya[2] = {k1.sign ? ny : ys, k1.sign ? ys : ny};  // by d < 0
    const Fq* const yb[2] = {k2.sign ? ny : ys, k2.sign ? ys : ny};
    // Storage for the sum: add_entry() constructs it at the first addition, which the top digit
    // makes (top > 0) before anything reads it. A default-constructed Point would spend a run-time
    // Montgomery conversion on a y of 1 that nothing reads.
    alignas(32) std::byte sum_raw[sizeof(Point)];
    auto& sum = *reinterpret_cast<Point*>(sum_raw);
    bool started = false;  // Unconstructed sum, or infinity after a cancellation: no doubling.
    for (auto i = top; i-- != 0;)
    {
        if (started)
            dbl_inplace(sum);
        if (const int d = naf_a[i]; d != 0)
            add_entry(sum, started, d, xs, ya);
        if (const int d = naf_b[i]; d != 0)
            add_entry(sum, started, d, bx, yb);
    }
    sum.z *= z;  // Back from the image: z == 0 (the point at infinity) stays 0.
    return sum;
}
}  // namespace
#endif

AffinePoint mul(const AffinePoint& pt, const uint256& c) noexcept
{
    if (pt == 0)
        return pt;

    if (c == 0)
        return {};

#if EVMONE_BN254_MUL_WNAF
    return ecc::to_affine(mul_wnaf(pt, c));
#else
    // Optimized using field endomorphism with scalar decomposition.
    // See ecc::decompose() for more details.
    const auto [k1, k2] = ecc::decompose<Curve>(c);

    const auto q = AffinePoint{Curve::BETA * pt.x, k2.sign ? -pt.y : pt.y};
    const auto p = AffinePoint{pt.x, k1.sign ? -pt.y : pt.y};
    const auto pr = msm(k1.value, p, k2.value, q);
    return ecc::to_affine(pr);
#endif
}
}  // namespace evmone::crypto::bn254
