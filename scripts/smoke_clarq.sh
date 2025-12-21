#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-smoke}"
OUTDIR="artifacts/clarq/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

CMD=(bash -lc "cd external/clarq-llm && ls -la && test -f evaluation.sh && echo 'evaluation.sh found'")

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"
("${CMD[@]}" |& tee "$OUTDIR/stdout.log") || true

echo "Wrote: $OUTDIR"
