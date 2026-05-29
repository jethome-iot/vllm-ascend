# Upstream contribution candidates

**Status:** Skeleton (SP-0a A0a.8 placeholder). SP-1+ populates с cherry-pick descriptions.

## Policy (per master plan §11.4 «Upstream contribution sync» v2)

**No upstream PR submissions** to `vllm-project/vllm-ascend` until:
1. SP-7 (Production) validation complete
2. Production soak ≥ 1 week green
3. Explicit user OK для submission

Until that gate, this directory accumulates **cherry-pick-ready PR descriptions**.

## Per-candidate file format

```markdown
# Upstream candidate: <short title>

**Source sub-plan:** SP-X
**Status:** ready, not submitted (per master §11.4 v2)
**Target upstream issues:** #ISSUE_NUM, #ISSUE_NUM
**Submission gate:** Post-A7 production validation + user OK

## Description
<PR-ready description с rationale, test coverage, before/after>

## Cherry-pick procedure
<when ready: `git cherry-pick <commit-hash-1> <commit-hash-2>`>

## Validation in our fork
<tests passing, soak periods, prod observations>
```

## Expected candidates (over SP-1..SP-7)

- `decode-fixes.md` — fixes для #7306 (MambaSpec dtype) + #7109 (FP32 ssm_state) — SP-1 A1a
- `qwen3_5_model.md` — Qwen3.5 hybrid model definition — SP-2 stepping stone
- `chunk_gated_delta_rule_fwd_310p.md` — naive Ascend C prefill kernel — SP-3
- `chunk_gated_delta_rule_fwd_optimized.md` — Cube-optimized version — SP-5
