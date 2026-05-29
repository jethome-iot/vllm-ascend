# jh-vllm-ascend Architecture (jethome-iot fork)

**Status:** Living document — single source of truth for fork design.
**Owner:** adeep
**Last update:** 2026-05-30 (SP-0a A0a foundational docs)

---

## Purpose

This fork (`jethome-iot/vllm-ascend`) adds support for **Qwen3.6** hybrid Gated DeltaNet
(non-MoE 27B + MoE 35B-A3B) on Huawei Ascend 310P3 hardware. Base upstream:
`vllm-project/vllm-ascend` tag `v0.19.1rc1` (Docker image `v0.19.1rc1-310p`).

For full design context see:
- `huawei-ascend/docs/specs/2026-05-29-qwen36-vllm-ascend-310p3-design.md` (design spec v2)
- `huawei-ascend/docs/plans/2026-05-29-qwen36-implementation-plan.md` (master plan v3)
- `huawei-ascend/docs/plans/sub/SP-*.md` (8 sub-plans)

---

## Fork strategy

### Branches

| Branch | Purpose | Protection |
|---|---|---|
| `upstream/main` | Read-only mirror of `vllm-project/vllm-ascend@main`. Synced via PR (no direct write). | enforce_admins=true; no force-push; no admin bypass |
| `jh/main` | Our production branch. Periodically rebased on `v0.19.1rc1` / future tags. | 1 PR review required; admin override enabled (emergency); force-push admin only |
| `jh/qwen36-310p` | Long-lived feature branch. Discrete sub-phase PRs merge here, then promoted to `jh/main`. | Inherits `jh/main` policy |
| `jh/qwen36-310p/<sub-phase>-<desc>` | Short-lived feature branches per sub-phase (e.g., `A0a-foundational-docs`, `A2.1-cumsum-naive`). PR → `jh/qwen36-310p`. | None (working branches) |

### PR strategy (per user instruction 2026-05-30)

**No direct pushes to protected branches.** Every significant chunk of work:

1. Create feature branch: `jh/qwen36-310p/<sub-phase>-<short-desc>`
2. Atomic commits on feature branch
3. Internal PR → target branch (`jh/qwen36-310p` or `jh/main`)
4. CI green + self-review (+ Codex/Detective review where applicable)
5. Squash-merge for small PRs; merge commit for multi-file features

**No upstream PR submissions** to `vllm-project/vllm-ascend` until A7 production validation
+ user OK. `docs/upstream-candidates/*.md` accumulate cherry-pick descriptions для future
submission, but are not submitted automatically.

### Upstream tracking

- Daily quick scan (5 min): `git fetch upstream main && git log --oneline upstream/main..upstream/main@{1.day.ago} -- vllm_ascend/_310p/ vllm_ascend/models/ vllm_ascend/ops/`
- Weekly Monday deep dive: review critical PR/issue list (см. `docs/jh-upstream-status.md`)
- Bi-weekly: rebase `jh/main` on upstream; rebase feature branches на `jh/main`
- Reference: master plan §14.6 «Upstream tracker hygiene» v2

---

## Fork boundary

Defines what's ours vs upstream and what patch policy applies. Enforced via CI lint
`scripts/lint-sp0a-non-goals.sh` for SP-0a-active directories.

### Tier 1: Pure upstream mirror (no edits ever)

- `vllm_ascend/core/` — untouched, rebases must apply cleanly
- `vllm/` — vllm package itself, never edit
- `tests/upstream/` — upstream tests, must pass on every rebase

**Tier 1 violation:** CI rejects PR.

### Tier 2: Patched (small surgical changes, cherry-pickable upstream)

- `vllm_ascend/_310p/` — 310P-specific paths; our patches are decode fixes (#7306, #7109)
  - Policy: each change must be cherry-pickable as separate commit
- `vllm_ascend/patches/` — monkey-patches; tier-2 because they're temporary
  - Policy: patch file deleted when upstream merges fix

**Tier 2 policy:** Commit message must include rationale + link to upstream issue/PR.

### Tier 3: Our additions (jh-exclusive, no upstream contract)

- `vllm_ascend/ops/gdn/` — our GDN custom ops; eventual upstream candidate
- `vllm_ascend/models/qwen3_5.py` + `qwen3_6.py` + `qwen3_6_moe.py`
- `ops/` (Ascend C kernels) — native kernels, our build pipeline
- `docs/jh-*` — all jh-prefixed docs
- `memory/` — our atoms (sourced separately в huawei-ascend repo; not part of fork)

**Tier 3 policy:** Free, follows our conventions; no upstream contract obligation.

---

## Knowledge artifact pattern

Stated rule for where information lives. Prevents docs/memory duplication.

| Path | Purpose | Audience | Lifecycle |
|---|---|---|---|
| `docs/jh-architecture.md` | Single source of truth для fork design (this file) | engineers + agents | edited continuously |
| `docs/jh-decisions/*.md` | Reversible decisions per schema (`_template.md`) | engineers + agents | append-only; status changes (proposed → accepted → superseded) |
| `docs/jh-stack-switch-procedure.md` | MindIE↔vllm switch ops runbook (used by SP-0b + SP-1+ NPU windows) | engineers | edited rarely; tested via dry-run |
| `docs/jh-upstream-status.md` | Weekly upstream tracking snapshots + daily scan log | engineers | append-only weekly |
| `docs/abi/*.md` | Custom op ABI contracts | engineers + future contributors | append-only per op; SP-1+ populated |
| `docs/upstream-candidates/*.md` | Future PR descriptions, deferred submission | engineers | drafted continuously; submitted post-A7 user OK |
| `memory/jh-*.md` | (lives in huawei-ascend repo, not fork) Agent-readable working notes | agents | append-only; supersedes via `[[link]]` |

**Boundary rule:** Quick capture → `memory/` (in huawei-ascend repo). When crystallized
into durable, externally-citable knowledge → migrate to `docs/jh-*` (in this fork).
Decisions ALWAYS in `docs/jh-decisions/` per schema, never in `memory/`. Operational
runbooks (switch procedure, recovery) in `docs/` after dry-run testing.

---

## Version freeze

Frozen tuple of versions all SP-0+ work builds against. Updated when a coordinated
rebase happens (e.g., после A4 driver upgrade).

### Build-time (from image `jh-stable` inspection)

- CANN: *Pending A0a.11 image inspection*
- torch_npu: *Pending A0a.11 image inspection*
- vllm-ascend tag: `v0.19.1rc1` (commit `da421afad7192dac64e39ae1d32305d57344f3cf`)
- Base ARM64 image digest: *Pending A0a.10 CI build summary*
- fw version: *Pending SP-0b A0b watchdog (npu-smi info during runtime)*

### Decided separately (from external sources)

- **FLA (FlashLinearAttention) tag:** `v0.5.0` (commit `3a9ce1c83a13994d824dbb3421e2989d330bb38b`)
  - Rationale: tagged release more stable than rolling `main`. Main HEAD recorded as
    `b43408e002c90ad7260ae6214c0b0d0ac189a49f` for reference (2026-05-30 captured).
- **Qwen3.5-9B HF:** commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (lastModified 2026-03-02)
- **Qwen3.6-27B HF:** commit `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (lastModified 2026-04-24)
- **Qwen3.6-35B-A3B HF:** commit `995ad96eacd98c81ed38be0c5b274b04031597b0` (lastModified 2026-04-24)
- **Eco-Tech baseline (Qwen3-32B-w8a8sc-310-vllm) ModelScope:**
  commit `bf07011d56ac511e2b55dee9ef9970febd361256` (master branch)

### Update policy

Version freeze updated via decision artifact (`docs/jh-decisions/<date>-version-freeze-bump.md`)
when:
- Upstream rebase moves to new tagged release
- HF model SHA changes detected via daily scan и mainstream adoption confirmed
- A4 driver upgrade requires CANN/torch_npu bump

---

## Repository layout

```
jh-vllm-ascend/                        # this fork (jethome-iot/vllm-ascend)
├── vllm_ascend/                       # upstream + our additions
│   ├── core/                          # Tier 1 (pure mirror)
│   ├── _310p/                         # Tier 2 (patched)
│   ├── patches/                       # Tier 2 (monkey-patches)
│   ├── models/                        # Tier 3 (qwen3_5.py + qwen3_6.py + qwen3_6_moe.py — SP-2/SP-4)
│   └── ops/
│       └── gdn/                       # Tier 3 (custom GDN ops — SP-1+)
├── ops/                               # Tier 3 (Ascend C kernel sources — SP-3+)
├── docs/
│   ├── jh-architecture.md             # this file
│   ├── jh-decisions/                  # decision artifacts (schema in _template.md)
│   ├── jh-stack-switch-procedure.md   # MindIE↔vllm switch runbook
│   ├── jh-upstream-status.md          # upstream tracking
│   ├── abi/                           # custom op ABI contracts (SP-1+)
│   └── upstream-candidates/           # cherry-pick PR descriptions (post-A7 submission)
├── tests/
│   ├── upstream/                      # Tier 1 mirrored upstream tests
│   ├── ops/                           # our op-level tests (SP-1+)
│   └── fixtures/                      # golden NPZ via Git LFS (SP-1+)
├── scripts/
│   ├── lint-decisions.sh              # CI lint для decision schema
│   ├── lint-sp0a-non-goals.sh         # CI lint preventing premature stubs
│   └── regen-runner-token.sh          # ARM64 runner token re-registration
└── .github/
    └── workflows/
        ├── jh-build.yml               # main CI image build
        ├── hello-runner.yml           # runner connectivity test
        └── upstream-sync.yml          # PR-based mirror sync (no direct write)
```

---

## Related external knowledge

- `huawei-ascend/AGENTS.md` — hardware constraints, gotchas, server access discipline
- `huawei-ascend/memory/atlas-300i-duo-tp-cross-card-kernel-hang.md` — driver-related known hang
- `huawei-ascend/memory/eco-tech-w8a8sc-310-inventory.md` — pre-quantized weights catalog
- `huawei-ascend/memory/vllm-ascend-310p3-support-matrix.md` — validated capabilities
