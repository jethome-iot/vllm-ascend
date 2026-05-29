# Decision artifacts

**Status:** Ready (SP-0a A0a.6 created).

Decision documents capture **reversible** technical/operational decisions с explicit
rationale, consequences, reversibility, и consumers.

## How to use

1. Copy `_template.md` to `<YYYY-MM-DD>-<short-slug>.md`
2. Fill **all** required sections (CI lint `scripts/lint-decisions.sh` enforces)
3. Open PR (per `docs/jh-architecture.md` PR strategy)
4. Status flow: `proposed` → `accepted` (после approval) → `superseded` (when replaced)

## Required headers (CI-enforced)

```
^# Decision:
^\*\*Date:
^\*\*Status:
^\*\*Owner:
^## Context
^## Options considered
^## Constraints
^## Chosen
^## Consequences
^## Reversibility
^## Consumers
```

## Examples (expected over SP-0a..SP-7)

- `<date>-image-registry.md` — local Harbor vs GHCR (SP-0a P1.5)
- `<date>-fla-reference-machine.md` — Apple Silicon M4 Ultra (SP-0a P2.1, already decided)
- `<date>-fla-commit-pin.md` — v0.5.0 vs main HEAD (SP-0a P4.2)
- `<date>-mamba-state-fp32.md` — always FP32 для GDN ssm_state (SP-1 A1a)
- `<date>-version-freeze-bump.md` — when upstream rebase moves baseline
