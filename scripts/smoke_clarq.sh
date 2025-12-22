#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-smoke}"
OUTDIR="artifacts/clarq/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# Smoke = verify repo structure + python imports (no API keys, no evaluation run)
CMD=(bash -lc "cd external/clarq-llm && \
  ls -la evaluation.sh evaluation.py l2l.py >/dev/null && \
  python -c \"import evaluation; import l2l; print('ClarQ imports OK')\"")

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"
("${CMD[@]}" |& tee "$OUTDIR/stdout.log")

echo "Wrote: $OUTDIR"
