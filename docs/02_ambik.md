# AmbiK Experiments

This document describes how AmbiK experiments are run and evaluated within the paper repository.

AmbiK experiments are executed using the `ambik_evaluation` codebase, which is included as a git submodule under:

external/ambik_evaluation/

The paper repository provides wrapper scripts and a standardized artifact structure for reproducibility.

---

## Environment Setup

AmbiK experiments require a dedicated Python environment due to dependencies on:
- PyTorch
- Hugging Face Transformers
- SAE tooling (`sae-lens`, `transformer-lens`)
- Sentence embeddings and NLI models

We recommend using a separate Conda environment.

Example:

```bash
conda create -n ambik python=3.10
conda activate ambik
pip install -r external/ambik_evaluation/requirements.txt
```

## Running evaluation

Without steering:

```python
    python external/ambik_evaluation/runner.py \
    --model_name google/gemma-2b-it \
    --num_examples 2 \
    --seed 0 \
    --mode proxy \
    --out_json artifacts/ambik/example_run/results.json
```

With steering:

```python
    python runner.py \
        --model_name gemma \
        --use_steering \
        --steering_features "771, 344, 383, 688, 565, 591" \
        --sae_release gemma-scope-9b-it-res-canonical \
        --sae_id layer_20/width_16k/canonical \
        --gemma_model gemma-2-9b-it \
        --seed 50 \
        --steering_strength 3 \
        --max_act 4 \
        --num_examples 100 \
        --mode proxy \
        --dataset_csv data/ambik_calib_100.csv \
        --out_json results/ambik_9b_steered_eval.json
```

Key options:

* `--model_name`: LLM used for question generation (Gemma is the primary model used in this work)
* `--mode`: proxy, dialog, or both
* `--num_examples`: subsampling for debugging or development
* `--use_steering`: enable SAE-based steering
* `--steering_features`, `--steering_strength`, `--max_act`: steering configuration