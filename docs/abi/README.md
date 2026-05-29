# Custom op ABI contracts

**Status:** Skeleton (SP-0a A0a.8 placeholder). SP-1+ populates ABI contracts.

<!-- DO NOT POPULATE UNTIL SP-1 -->

## Planned contents

Per `huawei-ascend/docs/specs/2026-05-29-qwen36-vllm-ascend-310p3-design.md` §3:

- `chunk_gated_delta_rule_fwd_310p.md` — main prefill kernel ABI (SP-1 A1b finalized;
  SP-3 refinements via decision artifacts)
- `chunk_local_cumsum_310p.md` — cumsum helper (SP-1 skeleton; SP-3 full)
- `recompute_w_u_fwd_310p.md` — recompute helper
- `wy_fast_fwd_310p.md` — wy_fast helper
- `chunk_fwd_h_310p.md` — chunk forward state buildup
- `chunk_output_kernel_310p.md` — output gather

Each ABI doc per spec §3.1 template includes:
- Inputs (shape/dtype/role)
- Outputs (shape/dtype/role)
- Invariants (stride layout, stream model, error semantics)
- Error codes (aclError mapping)
- Versioning (schema version, implementation status)
