# intx (vendored)

This is a vendored copy of the [intx](https://github.com/chfast/intx) extended
precision integer library — only the single public header `include/intx/intx.hpp`,
plus a minimal `CMakeLists.txt` exposing the upstream `intx::intx` target.

Snapshot: `main` @ [`88753c7bc97cdd9075cc66302aa89ebd01c6718e`](https://github.com/chfast/intx/commit/88753c7bc97cdd9075cc66302aa89ebd01c6718e) (2026-10-13).

To update, replace `include/intx/intx.hpp` (and `LICENSE`) with the versions
from the desired upstream commit and record the new revision above.

## Airbender rv32 changes

On top of the snapshot, `include/intx/intx.hpp` carries the Airbender rv32im
changes from [erigontech/intx](https://github.com/erigontech/intx)
`release/airbender` (up to `852b11ed`), rebased onto `88753c7`. Most are
behind `__riscv_xlen == 32` / `AIRBENDER` guards; `uint`'s constructors zero
its words explicitly (for `uninit_tag`) on every target. When updating the
snapshot, rebase those commits again.
