# GDN custom ops (Gated DeltaNet) для Ascend 310P3

**Status:** Skeleton (SP-0a A0a.8 placeholder). SP-1+ populates with Python wrappers
для Ascend C kernels.

<!-- DO NOT POPULATE UNTIL SP-1 -->

## Planned contents (SP-1+)

- `__init__.py` — Python module marker
- `chunk_gated_delta_rule_fwd.py` — wrapper around Ascend C op (SP-1 stub, SP-3 real)
- `fused_recurrent_gated_delta_rule.py` — wrapper around torch_npu op with FP32 fix (SP-1)
- `chunk_local_cumsum.py` — wrapper для cumsum helper (SP-2+)
- `recompute_w_u_fwd.py` — wrapper для recompute helper (SP-2+)
- `wy_fast.py` — wrapper для wy_fast helper (SP-2+)

CI enforces premature stub prevention via `scripts/lint-sp0a-non-goals.sh` while SP-0a active.
