# Interpretable Steering for Clarification Behavior in LLMs

This repository contains the code, scripts, and documentation used to run the experiments reported in the paper "Steering Large Language Models Toward Clarification through Sparse Autoencoders".

It unifies three previously independent codebases into a single, reproducible workflow:
- AmbiK (ambiguity resolution and clarification prompting)
- ClarQ-LLM (dialog-based clarification benchmarks)
- SAE-based steering for Gemma models

This repository includes three external codebases as git submodules:

- **AmbiK** — used for single-turn ambiguity resolution and clarification experiments  
- **ClarQ-LLM** — used for multi-turn clarification dialogue experiments  
- **SAE-Reasoning** — used to **compute feature scores (e.g., ReasonScore)** over SAE features and generate dashboards for inspecting top features. We use these scores to choose which SAE features to steer in downstream experiments (AmbiK / ClarQ-LLM).

---

## Method Overview

![Method overview](figures/meth_od.png)

**High-level idea:**  
We steer a local LLM (Gemma) using interpretable Sparse Autoencoder (SAE) features to control *when and how* the model asks clarification questions. The steered model is evaluated across two complementary benchmarks:

1. **AmbiK** — single-turn ambiguity resolution
2. **ClarQ-LLM** — multi-turn clarification dialogues

The provider model (Qwen) supplies information when clarification is requested, while evaluation is performed using official benchmark scripts.

---

## Repository Structure

```
clarifySAE-steering/
├── external/
│   ├── ambik_evaluation/        # AmbiK repo (submodule)
│   ├── clarq-llm/               # ClarQ-LLM repo (submodule)
│   └── SAE-Reasoning/           # SAE-Reasoning repo (submodule)
│
├── docs/
│   ├── 01_ambik.md              
│   ├── 02_clarq_llm.md          
│   └── 03_sae_reasoning.md      # feature selection 
│
├── envs/
│   ├── ambik.yml                # AmbiK environment
│   ├── clarq.yml                # ClarQ-LLM environment
│   └── sae_reasoning.yml        # SAE-Reasoning environment 
│
├── scripts/
│   ├── _run_record.sh
│   ├── smoke_ambik.sh
│   ├── smoke_clarq.sh
│   ├── smoke_sae_reasoning.sh  
│   ├── run_clarq_gemma_seeker_qwen_provider.sh
│   └── run_clarq_eval.sh
│
├── artifacts/
│   ├── ambik/
│   ├── clarq/
│   └── sae_reasoning/           
│
├── figures/
│   └── method_overview.png
│
└── README.md

```
