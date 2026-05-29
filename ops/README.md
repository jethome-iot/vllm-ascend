# Native Ascend C kernel sources

**Status:** Skeleton (SP-0a A0a.8 placeholder). SP-3+ populates с naive kernels;
SP-5 optimizes (Cube units, pipelining).

<!-- DO NOT POPULATE UNTIL SP-1 (ABI stub) / SP-3 (real implementation) -->

## Planned contents

Per `docs/abi/*.md` ABI contracts:

- `chunk_gated_delta_rule_fwd_310p/` — main prefill kernel (SP-3)
- `chunk_local_cumsum_310p/` — cumsum helper (SP-3)
- `recompute_w_u_fwd_310p/` — recompute helper (SP-3)
- `wy_fast_fwd_310p/` — wy_fast helper (SP-3)
- `chunk_output_kernel_310p/` — output gather (SP-3)

Each kernel dir contains:
- `kernel.cpp` — Ascend C kernel code (target `ascend310p1`)
- `host.cpp` — AscendCL host wrapper
- `CMakeLists.txt` — build config
- `README.md` — per-kernel notes

Build pipeline: `docs/build-ops-310p.md` (SP-1 A1b deliverable).

CI enforces premature stub prevention via `scripts/lint-sp0a-non-goals.sh` while SP-0a active.
