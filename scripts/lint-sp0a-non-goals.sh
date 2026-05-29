#!/bin/bash
# Lint: SP-0a non-goals enforcement.
# Per SP-0a-bootstrap-infra.md v3 §1 «Explicit non-goals».
#
# Disabled when tag `sp1-v1-done` exists (SP-1 has started populating ops/gdn, abi).
#
# Forbidden during SP-0a active:
#   vllm_ascend/ops/gdn/* (except README.md)
#   ops/* (except README.md per-dir, and per-subdir README.md)
#   docs/abi/* (except README.md)

set -euo pipefail

# Check if SP-1 has started — disable this lint if so
if git tag -l "sp1-v1-done" | grep -q "^sp1-v1-done$"; then
  echo "✓ SP-1 done — SP-0a non-goals lint disabled"
  exit 0
fi

FORBIDDEN_DIRS=("vllm_ascend/ops/gdn" "ops" "docs/abi")

violations=()
for dir in "${FORBIDDEN_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    # Find non-README.md files
    while IFS= read -r -d '' f; do
      violations+=("$f")
    done < <(find "$dir" -type f ! -name 'README.md' -print0 2>/dev/null)
  fi
done

if [ ${#violations[@]} -gt 0 ]; then
  echo "❌ SP-0a non-goal violation: premature stubs detected in forbidden dirs"
  for v in "${violations[@]}"; do
    echo "   $v"
  done
  echo ""
  echo "These directories should be empty (only README.md placeholders) until SP-1 starts."
  echo "If you are SP-1 and ready to populate, ensure git tag sp1-v1-done is created first."
  exit 1
fi

echo "✓ SP-0a non-goals respected"
exit 0
