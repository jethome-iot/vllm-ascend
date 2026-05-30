# Upstream tracking status

**Status:** Initial entry (SP-0a A0a.12 deliverable).
**Cadence:** Daily quick scan (in `huawei-ascend/memory/upstream-problem-areas.md`),
weekly Monday deep dive (this doc).

Per master plan §14.6 «Upstream tracker hygiene» v2:
- Daily: `git fetch upstream main && git log --oneline upstream/main..upstream/main@{1.day.ago} -- vllm_ascend/_310p/ vllm_ascend/models/ vllm_ascend/ops/`
- Weekly: critical PR/issue audit (this file)
- Bi-weekly: rebase `jh/main` на upstream; rebase feature branches на `jh/main`

---

## 2026-05-30 — Initial entry

**Baseline:** `v0.19.1rc1-310p` (Docker tag), git commit `da421afad7192dac64e39ae1d32305d57344f3cf`
upstream/main HEAD at fork moment: `2a77209accf9a49845716cf79d4227bb9237ef32`

### Critical PR/issue tracking list (current status per fork creation)

| Ref | Title | Status |
|---|---|---|
| [#7394](https://github.com/vllm-project/vllm-ascend/issues/7394) | Qwen3.5/MoE/GDN/conv1d/W8A8 на 300I Duo (master tracker) | open, in progress |
| [#7306](https://github.com/vllm-project/vllm-ascend/issues/7306) | Qwen3.5-9B 310P3 MambaSpec dtype | open (target SP-1 A1a) |
| [#7398](https://github.com/vllm-project/vllm-ascend/pull/7398) | fused recurrent gated delta rule PyTorch fallback + 310P UT | merged (used as baseline) |
| [#7798](https://github.com/vllm-project/vllm-ascend/pull/7798) | `[310p] npu_causal_conv1d_310 AscendC Custom Op` | merged (our binding reference for SP-1 A1b) |
| [#7109](https://github.com/vllm-project/vllm-ascend/pull/7109) | patch_qwen3_5 FP32 ssm_state fix | merged (reference for SP-1 A1a) |
| [#3017](https://github.com/vllm-project/vllm-ascend/issues/3017) | Qwen3-Next 310P root tracker | open |
| [#1565](https://github.com/vllm-project/vllm-ascend/issues/1565) | TP=4 crash 310P | open (SP-5 territory) |
| [#1700](https://github.com/vllm-project/vllm-ascend/pull/1700) | PP V1 engine | merged (our PP=2×TP=2 fallback path) |
| [#5318](https://github.com/vllm-project/vllm-ascend/issues/5318) | Q1 2026 Roadmap | open (reference для feature priority) |

### Our position relative to upstream

- **No upstream PR submissions** until A7 production validation + user OK (master plan §11.4 v2)
- `docs/upstream-candidates/` будет накапливать cherry-pick descriptions
- Daily scan активен по schedule в `huawei-ascend/memory/upstream-problem-areas.md`

### Next weekly review

Suggested next deep dive: **2026-06-01 (Monday)**. Items для audit:
- New commits на `_310p/`, `models/`, `ops/` за неделю
- Any of critical tracking list issues closed?
- New 310p-related issues opened?
- Rebase candidate: `jh/main` against `upstream/main` HEAD
