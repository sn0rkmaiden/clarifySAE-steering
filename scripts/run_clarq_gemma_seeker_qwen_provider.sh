#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-gemma_seeker_qwen_provider}"
EVAL_SET="${2:-0-0}"          # keep tiny by default
TASK_DATA="${3:-data/English}" # relative to external/clarq-llm

if [[ -z "${NEBIUS_API_KEY:-}" ]]; then
  echo "Set NEBIUS_API_KEY in your shell (provider Qwen uses it)."
  exit 1
fi

OUTDIR="artifacts/clarq/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

CMD=(bash -lc "
  set -e
  cd external/clarq-llm
  python l2l.py \
    --seeker_agent_llm gemma \
    --provider_agent_llm qwen \
    --task_data_path '${TASK_DATA}' \
    --evaluation_set '${EVAL_SET}' \
    --hftoken '${NEBIUS_API_KEY}'
")

./scripts/_run_record.sh "$OUTDIR" "${CMD[@]}"
("${CMD[@]}" |& tee "$OUTDIR/stdout.log")

# Collect outputs (copy the newest l2l_gemma*.json into artifacts)
mkdir -p "$OUTDIR/results_snapshot"
LATEST=$(ls -t external/clarq-llm/results/l2l_gemma*.json 2>/dev/null | head -n 1 || true)
if [[ -n "$LATEST" ]]; then
  cp "$LATEST" "$OUTDIR/results_snapshot/"
  echo "Copied output: $LATEST"
else
  echo "Warning: No l2l_gemma*.json found in external/clarq-llm/results/"
fi

echo "Wrote: $OUTDIR"
