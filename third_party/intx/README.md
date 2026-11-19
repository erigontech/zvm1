# intx (vendored)

This is a vendored copy of the [intx](https://github.com/chfast/intx) extended
precision integer library — only the single public header `include/intx/intx.hpp`,
plus a minimal `CMakeLists.txt` exposing the upstream `intx::intx` target.

Snapshot: `main` @ [`c993582a3f99f4ebb889cf64e1a97e0fbcdfed25`](https://github.com/chfast/intx/commit/c993582a3f99f4ebb889cf64e1a97e0fbcdfed25) (2026-11-19).

To update, replace `include/intx/intx.hpp` (and `LICENSE`) with the versions
from the desired upstream commit and record the new revision above.

## Airbender rv32 changes

On top of the snapshot, `include/intx/intx.hpp` carries the Airbender rv32im
changes from [erigontech/intx](https://github.com/erigontech/intx)
`release/airbender` (up to `852b11ed`), rebased onto `88753c7` and merged
into the update to `c993582`. Most are behind `__riscv_xlen == 32` /
`AIRBENDER` guards; `uint`'s constructors zero its words explicitly (for
`uninit_tag`) on every target. When updating the snapshot, rebase those
commits again.

## RISC-V comparisons

On RISC-V (both rv32 and rv64) the `uint` comparisons go by machine words
(`size_t`): `uint < uint` from the top word down to the first that differs,
and the comparisons with builtin integers by checking the words above the
integer's for zero, then comparing the low words. RISC-V has no carry flag, so
upstream's `subc()` borrow chain and its branch-free comparisons with builtin
integers take more instructions.
