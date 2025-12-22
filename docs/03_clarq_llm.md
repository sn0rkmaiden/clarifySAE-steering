
# ClarQ-LLM Experiments

This document describes how ClarQ-LLM experiments are generated and evaluated in the paper repository.

The ClarQ-LLM codebase is included as a git submodule under:

external/clarq-llm/

We distinguish between:
1. **Generation** (running the seeker/provider interaction)
2. **Evaluation** (computing ClarQ metrics)

---

## Environment Setup

ClarQ-LLM has a dependency profile distinct from AmbiK and should be run in a separate environment.

Example:

```bash
conda create -n clarq python=3.10
conda activate clarq
pip install -r external/clarq-llm/requirement.txt
```

## Running dialogues

ClarQ generation is performed using `l2l.py`.

```python
export NEBIUS_API_KEY="..."
bash scripts/run_clarq_gemma_seeker_qwen_provider.sh test 1-5
```

## Evaluation

ClarQ metrics are computed using the evaluation script `external/clarq-llm/evaluation.py`. Evaluation uses an LLM-based judge (Qwen in our setup) and therefore requires an API key.

```bash
bash scripts/run_clarq_eval.sh <run_name> <l2l_json> <evaluation_set>
```

Example:
```python
export NEBIUS_API_KEY="..."
bash scripts/run_clarq_eval.sh \
  gemma_comp_en_smoke \
  external/clarq-llm/results/l2l_gemma.Comp.En.json \
  3-6
```

