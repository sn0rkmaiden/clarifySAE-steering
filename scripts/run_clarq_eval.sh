#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?run_name required}"
L2L_JSON="${2:?path to l2l json required}"
EVAL_SET="${3:?evaluation_set required}"

if [[ -z "${NEBIUS_API_KEY:-}" ]]; then
  echo "Set NEBIUS_API_KEY before running ClarQ evaluation."
  exit 1
fi

# Allow either:
# - results/xxx.json (relative to external/clarq-llm)
# - external/clarq-llm/results/xxx.json
if [[ "$L2L_JSON" == external/clarq-llm/* ]]; then
  L2L_JSON="${L2L_JSON#external/clarq-llm/}"
fi

OUTDIR="artifacts/clarq/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

CMD=(bash -lc "
  set -e
  cd external/clarq-llm
  python evaluation.py qwen \"$L2L_JSON\" \"$NEBIUS_API_KEY\" \"$EVAL_SET\"
")

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"
("${CMD[@]}" |& tee "$OUTDIR/stdout.log")

echo "Wrote: $OUTDIR"
