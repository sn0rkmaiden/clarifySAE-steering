This repo pins three codebases as submodules and provides a single artifacts/ folder for outputs.

## Setup
git submodule update --init --recursive

## Smoke tests
bash scripts/smoke_ambik.sh
bash scripts/smoke_clarq.sh
bash scripts/smoke_steering.sh

Outputs land in artifacts/