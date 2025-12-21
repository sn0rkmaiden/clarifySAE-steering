# !/usr/bin/env bash
set -euo pipefail

# Usage: _run_record <artifact_dir> <command...>
ARTDIR="$1"; shift
mkdir -p "$ARTDIR"

# Record command + git state of submodules
printf "%s\n" "$*" > "$ARTDIR/command.txt"

{
  echo "=== DATE ==="
  date -Is
  echo
  echo "=== GIT (paper hub) ==="
  git rev-parse HEAD || true
  echo
  echo "=== SUBMODULES ==="
  git submodule status || true
  echo
  echo "=== PYTHON ==="
  python --version 2>&1 || true
  echo
  echo "=== PIP FREEZE (best effort) ==="
  python -m pip freeze 2>/dev/null || true
} > "$ARTDIR/env.txt"
EOF

chmod +x scripts/_run_record.sh
