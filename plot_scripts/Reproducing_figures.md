To reproduce figures from the paper run the following commands:

```python
# clone repo and pull submodules
git clone --recurse-submodules https://github.com/sn0rkmaiden/clarifySAE-steering.git

cd clarifySAE-steering

git submodule update --init --recursive
```

### Plots

* RQ1 Ambik main bars:

```python
python plot_scripts/plot_ambik_main_bars.py   --results data/results_without_rand.zip   --baseline_2b data/ambik_2b_baseline_featNone_new.json   --out_dir figures --baseline_9b data/results/9bmodel/question_vocab/results_layer20_question/ambik_9b_baseline_seed50_act4_featNone.json
```

* RQ3 Ambik line graph:

```python
python plot_scripts/plot_ambik_rq3_alpha_compare.py   --results data/results_without_rand.zip   --out_dir figures
```

* RQ3 Ambik features table:

```python
# go to SAE-Reasoning repo
cd external/sae-reasoning
```
```python
python plot_scripts/make_rq3_tables.py --root extraction/experiments  --k 50 --min_output_score 0 --topn 3
```

* Plot for ClarQ-LLM (outcome vs effort):

```python
python plot_scripts/clar_outcome_effort_from_metrics.py  --root data/ClarQ-LLM_results --out figures/clarq_effort_vs_outcome.pdf
```