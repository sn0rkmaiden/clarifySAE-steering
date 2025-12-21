#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-smoke}"
OUTDIR="artifacts/ambik/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# NOTE: Replace the command below with the smallest AmbiK command you actually use.
# The point is: "runs fast, produces some output, proves environment works".
CMD=(python external/ambik_evaluation/runner.py --help)

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"

# Run and log
("${CMD[@]}" |& tee "$OUTDIR/stdout.log") || true

echo "Wrote: $OUTDIR"
EOF

chmod +x scripts/smoke_ambik.sh
