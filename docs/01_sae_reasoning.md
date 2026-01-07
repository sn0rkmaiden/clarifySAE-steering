# SAE-Reasoning: feature scoring and selection

This repository is used to *rank SAE features* and decide **which feature IDs to steer** in downstream experiments (AmbiK / ClarQ-LLM).

In this paper repo we use SAE-Reasoning for three main tasks:

1) **Compute ReasonScore** for all SAE features given a vocabulary of "clarification" tokens.
2) **Compute OutputScore** for top-ranked features to estimate how strongly a feature can influence a model’s output distribution.
3) **Inspect OutputScores** (human-readable report: token strings, ranks, probabilities, etc.)

The submodule lives under:

external/SAE-Reasoning/

---

## Environment

SAE-Reasoning should be run in a separate environment. Example:

```bash
conda create -n sae_reasoning python=3.11
conda activate sae_reasoning
pip install -r external/SAE-Reasoning/requirements.txt
```