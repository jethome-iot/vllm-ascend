# Decision: <short title>

**Date:** YYYY-MM-DD
**Status:** proposed | accepted | superseded
**Owner:** <name>
**Affected sub-plans:** SP-X, SP-Y

## Context

<1-3 sentences on what triggered this decision>

## Options considered

- **Option A:** <description>
  - Pros: ...
  - Cons: ...
- **Option B:** <description>
  - Pros: ...
  - Cons: ...
- **Option C:** <description>
  - Pros: ...
  - Cons: ...

## Constraints

- <hard constraint 1, e.g., AGENTS.md «IPMI available»>
- <hard constraint 2>

## Chosen

**Option X**, because <rationale>.

## Consequences

- **Positive:** <what we get>
- **Negative:** <what we give up>
- **Neutral:** <side effects>

## Reversibility

- **Cost to revert:** low | medium | high
- **Trigger to revert:** <condition that would make us revisit>

## Consumers

- SP-X step Y.Z reads this decision (e.g., «SP-0a A0a.10 chooses image registry для CI push»)
- SP-W step A.B reads this decision

## Sources

- <links to relevant docs, issues, PRs>

---

*To use this template: copy to `<date>-<short-slug>.md`, fill all sections. CI lint
(`scripts/lint-decisions.sh`) verifies all required headers present.*
