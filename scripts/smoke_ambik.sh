#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-smoke}"
OUTDIR="artifacts/ambik/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

CMD=(
  python external/ambik_evaluation/runner.py
  --model_name google/gemma-2b-it
  --num_examples 2
  --seed 0
  --mode proxy
  --out_json "$OUTDIR/results.json"
)

mkdir -p "$OUTDIR"

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"

# Run and log
("${CMD[@]}" |& tee "$OUTDIR/stdout.log") || true

echo "Wrote: $OUTDIR"
