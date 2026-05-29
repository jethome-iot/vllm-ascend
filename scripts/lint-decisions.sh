#!/bin/bash
# Lint: docs/jh-decisions/*.md schema enforcement.
# Per master plan §0.4 decision schema.
# Required headers (regex anchored to line start):
#   ^# Decision:
#   ^\*\*Date:
#   ^\*\*Status:
#   ^\*\*Owner:
#   ^## Context
#   ^## Options considered
#   ^## Constraints
#   ^## Chosen
#   ^## Consequences
#   ^## Reversibility
#   ^## Consumers
# Skip _template.md (it's the schema itself).

set -euo pipefail

REQUIRED_HEADERS=(
  "^# Decision:"
  "^\*\*Date:"
  "^\*\*Status:"
  "^\*\*Owner:"
  "^## Context"
  "^## Options considered"
  "^## Constraints"
  "^## Chosen"
  "^## Consequences"
  "^## Reversibility"
  "^## Consumers"
)

DECISIONS_DIR="docs/jh-decisions"
if [ ! -d "$DECISIONS_DIR" ]; then
  echo "✓ No $DECISIONS_DIR directory — skip lint"
  exit 0
fi

violations=0
checked=0

for f in "$DECISIONS_DIR"/*.md; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  if [ "$base" = "_template.md" ] || [ "$base" = "README.md" ]; then
    continue
  fi
  checked=$((checked + 1))
  missing=()
  for hdr in "${REQUIRED_HEADERS[@]}"; do
    if ! grep -qE "$hdr" "$f"; then
      missing+=("$hdr")
    fi
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "❌ $f missing required headers:"
    for m in "${missing[@]}"; do echo "   $m"; done
    violations=$((violations + 1))
  fi
done

if [ "$violations" -gt 0 ]; then
  echo ""
  echo "FAIL: $violations decision files violate schema (checked $checked)"
  exit 1
fi

echo "✓ All $checked decision files conform to schema"
exit 0
