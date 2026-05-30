# Decision: SP-1 — rebase onto upstream + reduced scope

**Date:** 2026-05-30
**Status:** accepted (verified 2026-05-30 via Qwen3.5-9B end-to-end smoke)
**Owner:** adeep
**Affected sub-plans:** SP-1, SP-3 (deferred items)

## Context

При investigation SP-1 A1a.0 выявили, что upstream `vllm-project/vllm-ascend` уже merged
большинство fixes, которые мы планировали как нашу работу:

- #7109 (patch_qwen3_5 FP32) — MERGED 2026-03-10, уже в нашем baseline v0.19.1rc1
- #7398 (fused_recurrent PyTorch fallback + 310P UT) — MERGED 2026-03-25, уже в baseline
- #7306 (MambaSpec dtype) — OPEN, но **#9514 merged 2026-05-30** scopes MambaSpec
  construction в `model_runner_310p.py` + adds `patch_mamba_utils.py` (96 lines)
- Full PyTorch fallback for GDN (decode + chunk) — exists в baseline

Detail: `huawei-ascend/docs/investigations/2026-05-30-sp1-scope-shift.md`

## Options considered

### Option A: Cherry-pick #9514 surgically
- **Pros:** Minimal change; low risk regression; focused fix для #7306
- **Cons:** Stays на old baseline (v0.19.1rc1); other fixes (5 vulnerabilities, 110+ commits)
  remain unaddressed; еще больше divergence от upstream over time

### Option B: Rebase `jh/main` onto `upstream/main` HEAD
- **Pros:** All recent fixes pulled in (including #9514, security fixes, perf improvements);
  fork stays close to upstream — easier ongoing maintenance; clean linear history
- **Cons:** 110+ commits applied at once (more change surface); potential для unrelated
  regressions; force-push requires admin override

### Option C: Wait for next stable RC tag from upstream
- **Pros:** Maintainer-validated baseline
- **Cons:** Unknown ETA; blocks SP-1 indefinitely

### And SP-1 scope sub-decision:

#### Option X: Reduced SP-1 scope
- Skip ABI/stub kernel work (PyTorch fallback already в base, ABI contract можно делать в
  SP-3 на real signature basis)
- Focus только on: rebase application + verify Qwen3.5-9B end-to-end
- **Pros:** Lowest risk; actually-test reveals real issues; doesn't waste effort
- **Cons:** Less prepared для SP-3 если PyTorch fallback inadequate

#### Option Y: Original SP-1 scope
- Still write ABI contract + stub kernel pro forma
- **Pros:** Pre-builds SP-3 deliverables
- **Cons:** Speculative work; signatures may not match real Ascend C constraints

## Constraints

- AGENTS.md «никогда TP=4» (still valid)
- Driver state: only Card 1 visible (NPU 6), TP=2 single-card only
- BF16 model weights need FP16 patch (per AGENTS.md §«НЕ работали» 5)
- Master plan §11.4 v2: no upstream PR submissions
- Master plan §11.2 v2: feature-branch + PR workflow в fork

## Chosen

**Option B + Option X** = Rebase `jh/main` onto upstream/main + reduced SP-1 scope.

### Rationale

1. **Rebase** is cleaner fork maintenance strategy long-term. Our changes (docs/CI/workflows
   только) low conflict potential with upstream code changes.
2. **Reduced scope** acknowledges reality: PyTorch fallback exists, baseline regression
   test (Qwen3.5-9B) is the meaningful gate. ABI/stub work без real Ascend C constraints =
   speculative.
3. PR #9514 merged сегодня (2026-05-30) targets exactly наш use case (Mamba/GDN on 310p) —
   high value to adopt immediately.

## Consequences

### Positive
- jh/main now at upstream/main HEAD (commit `a8658e9d`); #9514 included
- All recent upstream fixes (security, perf, bugs) applied
- SP-1 calendar shrinks: 15 days median → **3-5 days realistic**
- ABI contract work moved to SP-3 where Ascend C constraints will be concrete

### Negative
- Force-push pre-rebase commit hashes broken для anyone with local clones
- Backup tag `pre-rebase-2026-05-30` (`732f7f87`) kept для recovery
- `upstream/main` mirror branch NOT updated (branch protection prevented direct
  force-push) — TODO: setup `upstream-sync.yml` workflow per master §11.4 reference
- ABI contract deferred → SP-3 has more upfront design work

### Neutral
- 5 vulnerabilities in upstream (3 critical, 2 moderate) — flagged by Dependabot;
  related to upstream deps, not our additions; tracked in jh-upstream-status.md
- Force-push на jh/main + jh/qwen36-310p обе синканы к `a8658e9d`

## Reversibility

- **Cost to revert:** Low — `git reset --hard pre-rebase-2026-05-30` returns to pre-rebase state
- **Trigger to revert:** Если новые upstream commits ломают наш baseline image build OR
  regress dense Qwen3-32B FP16 TP=2 performance (was 4.73 tok/s в SP-0b)

## Consumers

- **SP-1 A1a.1-A1a.4** — теперь mostly verification of #9514 fix, не writing patches
- **SP-1 A1b** — deferred; ABI doc moved to SP-3 А2.x
- **SP-3** — больше upfront work; will need real Ascend C kernel design from scratch
- **SP-4 model wiring** — same as before; Qwen3.5/3.6 wiring uses existing PyTorch fallback
- **CI rebuild required** — jh-build triggered by force-push, должен зелёным быть на новой
  baseline (Run #26679762160)

## Sources

- `huawei-ascend/docs/investigations/2026-05-30-sp1-scope-shift.md`
- Upstream PRs #7109, #7398, #7306, #9514
- Backup tag `pre-rebase-2026-05-30` → commit `732f7f87`
- Rebased jh/main HEAD: `a8658e9d`

## Verification result (added 2026-05-30 after A1a.1)

✅ Hypothesis confirmed: #7306 closed empirically by #9514 в нашем baseline+rebase.

**Test setup:**
- Image: `cr.jethome.work/jethome-iot/vllm-ascend:jh-stable` (post-rebase)
- Model: Qwen3.5-9B FP16, TP=2 single-card (chip 0+1 NPU 6), eager mode, max_model_len=4096
- Env: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (см. gotcha note)

**Result:**
- Engine init: 71.1s
- API server boot: ~30s after offline env vars
- 3 smoke prompts passed (RU greeting, EN math, EN code) — все coherent, ~6 tok/s
- Hybrid Mamba/GDN forward path stable (Triton/FLA prefill kernel)
- KV cache: 12.35 GiB → 199,680 tokens, concurrency 125× for 4096

**Gotcha discovered:** HuggingFace.co резолвится только в IPv6 на 10.183.1.25.
Docker host networking имеет broken IPv6 routing → SSL handshake hang в structured_output
init (calls `list_repo_files` → HF API). Solution: offline env vars in compose profile.
См. `huawei-ascend/memory/hf-hub-ipv6-ssl-hang-gotcha.md`.

**Outcome для A1a.2-A1a.4:**
- A1a.2 → no-op (#7109 already in baseline, verified via successful Qwen3.5 runtime)
- A1a.3 NPU UT → optional (PyTorch fallback works in production)
- A1a.4 upstream-candidates → no patches от нас для submission

См. `huawei-ascend/memory/sp1-a1a1-7306-empirically-closed.md` для подробного writeup.
