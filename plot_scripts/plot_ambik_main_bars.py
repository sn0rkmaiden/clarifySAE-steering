#!/usr/bin/env python3
import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_from_zip(z: zipfile.ZipFile, member: str) -> dict:
    with z.open(member) as f:
        return json.loads(f.read().decode("utf-8"))


def iter_jsons(results: Path):
    """Yield (name, obj) from either a zip file or a directory."""
    if results.is_file() and results.suffix.lower() == ".zip":
        with zipfile.ZipFile(results, "r") as z:
            members = [m for m in z.namelist() if m.lower().endswith(".json")]
            for m in members:
                try:
                    obj = load_json_from_zip(z, m)
                except Exception:
                    continue
                yield m, obj
    else:
        for p in results.rglob("*.json"):
            try:
                obj = load_json_file(p)
            except Exception:
                continue
            yield str(p), obj


def nanmean(xs: List[float]) -> float:
    xs2 = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, float) and math.isnan(x):
            continue
        xs2.append(x)
    return float(np.mean(xs2)) if xs2 else float("nan")


def is_steered(obj: dict) -> bool:
    return bool(obj.get("run_info", {}).get("steering_used", False))


def model_name(obj: dict) -> str:
    return obj.get("run_info", {}).get("model", {}).get("model_name", "")


def model_key(obj: dict, name_hint: str) -> Optional[str]:
    mn = model_name(obj).lower()
    if "2b" in mn:
        return "2B"
    if "9b" in mn:
        return "9B"
    s = name_hint.lower()
    if "2b" in s:
        return "2B"
    if "9b" in s:
        return "9B"
    return None


def infer_vocab(name_hint: str) -> Optional[str]:
    s = name_hint.lower()

    # 2B naming in results.zip
    if "results_2b_question" in s or "results_9b_question" in s:
        return "Q"
    if "results_2b_clar" in s or "results_9b_clar" in s:
        return "C"

    # 9B naming used elsewhere
    if "question_vocab" in s or "/question_vocab" in s:
        return "Q"
    if "clar_vocab" in s or "clar_vocab2" in s or "/clar_vocab" in s:
        return "C"

    return None



def compute_ambik_metrics(obj: dict) -> Dict[str, float]:
    """
    Computed over obj['examples'] (one element per AmbiK example):
      - clarif_rate = mean( num_questions > 0 )
      - acc         = mean( resolved_proxy )
      - avg_questions = mean( num_questions )
      - qsim        = mean( model_question_best_similarity )
    """
    ex = obj.get("examples", [])
    clar = [1 if e.get("num_questions", 0) > 0 else 0 for e in ex]
    acc = [1 if e.get("resolved_proxy", False) else 0 for e in ex]
    avgq = [float(e.get("num_questions", 0)) for e in ex]
    qsim = [e.get("model_question_best_similarity", float("nan")) for e in ex]
    return {
        "n": len(ex),
        "clarif_rate": nanmean(clar),
        "acc": nanmean(acc),
        "avg_questions": nanmean(avgq),
        "qsim": nanmean(qsim),
    }


def better_run(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Select the best steering run per (model,vocab).
    Primary: higher acc
    Tie-break: higher clarif_rate, then higher qsim.
    """
    if a["acc"] != b["acc"]:
        return a["acc"] > b["acc"]
    if a["clarif_rate"] != b["clarif_rate"]:
        return a["clarif_rate"] > b["clarif_rate"]
    return a["qsim"] > b["qsim"]


def find_9b_baseline(items, explicit_path: Optional[Path]) -> Tuple[str, dict]:
    if explicit_path is not None:
        return str(explicit_path), load_json_file(explicit_path)

    # Prefer an unsteered run with run_info.model.model_name == gemma-2-9b-it
    for name, obj in items:
        if is_steered(obj):
            continue
        if model_name(obj) == "gemma-2-9b-it":
            return name, obj

    # fallback heuristic
    for name, obj in items:
        if is_steered(obj):
            continue
        if "9b" in name.lower() and "baseline" in name.lower():
            return name, obj

    raise RuntimeError("Could not locate a 9B unsteered baseline in results. Pass --baseline_9b explicitly.")


# ----------------------------
# Plot styling
# ----------------------------
PALETTE = {
    "Baseline": "#E0D9D9",      
    "Steering (Q)": "#0046FF",  
    "Steering (C)": "#FF9013",  
}

# Font / layout knobs (paper-friendly)
FONT_BASE = 12
FONT_TICK = 12
FONT_TITLE = 15
FONT_LABEL = 13
FONT_LEGEND = 12
FONT_ANN = 11

# Increase title padding so it doesn't crowd the bars
TITLE_PAD = 20

# Lower => closer to figure bottom (reduces whitespace below "Model size")
SUPXLABEL_Y = 0.02



def nice_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def annotate(ax, bars):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(0.99, h + 0.02),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_ANN,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="Path to results.zip OR a directory containing AmbiK JSON runs.")
    ap.add_argument("--baseline_2b", type=Path, required=True,
                    help="Path to the fresh 2B baseline JSON.")
    ap.add_argument("--baseline_9b", type=Path, default=None,
                    help="Optional explicit 9B baseline JSON.")
    ap.add_argument("--out_dir", type=Path, default=Path("plots_out"))
    ap.add_argument("--out_name", type=str, default="ambik_main_bars_combined",
                    help="Base name for the output figure (pdf/png).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load 2B baseline (explicitly provided, since you fixed caching)
    base2b_obj = load_json_file(args.baseline_2b)
    base2b_m = compute_ambik_metrics(base2b_obj)

    # Load all items from results
    items = list(iter_jsons(args.results))

    # Find 9B baseline inside results unless explicit path provided
    base9b_name, base9b_obj = find_9b_baseline(items, args.baseline_9b)
    base9b_m = compute_ambik_metrics(base9b_obj)

    # Pick best steering run per (model, vocab)
    best: Dict[Tuple[str, str], Tuple[str, Dict[str, float]]] = {}
    for name, obj in items:
        if not is_steered(obj):
            continue
        vocab = infer_vocab(name)
        if vocab is None:
            continue
        mk = model_key(obj, name)
        if mk is None:
            continue
        met = compute_ambik_metrics(obj)
        key = (mk, vocab)
        if key not in best or better_run(met, best[key][1]):
            best[key] = (name, met)

    # Assemble a table (also saved as CSV for reproducibility)
    rows = [
        {"Model": "2B", "Method": "Baseline", **base2b_m, "source": str(args.baseline_2b)},
        {"Model": "9B", "Method": "Baseline", **base9b_m, "source": base9b_name},
    ]
    for mk in ["2B", "9B"]:
        for vocab, label in [("Q", "Steering (Q)"), ("C", "Steering (C)")]:
            key = (mk, vocab)
            if key not in best:
                rows.append({"Model": mk, "Method": label,
                             "n": float("nan"), "clarif_rate": float("nan"), "acc": float("nan"),
                             "avg_questions": float("nan"), "qsim": float("nan"),
                             "source": "NOT_FOUND"})
            else:
                name, met = best[key]
                rows.append({"Model": mk, "Method": label, **met, "source": name})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "ambik_main_bars.csv", index=False)

    # Print sources used (so you can cite exact configs)
    print("\n=== Sources used (exact files) ===")
    print(f"2B baseline: {args.baseline_2b}")
    print(f"9B baseline: {base9b_name}")
    print("\nBest steering runs (by Acc; ties -> clarif_rate -> qsim):")
    for k in sorted(best.keys()):
        name, met = best[k]
        print(f"  {k}: {name}  acc={met['acc']:.3f}  clarif_rate={met['clarif_rate']:.3f}  qsim={met['qsim']:.3f}")

    print("\n=== Aggregated values ===")
    print(df[["Model", "Method", "clarif_rate", "acc", "avg_questions", "qsim", "source"]].to_string(index=False))


    # Global matplotlib typography (paper-friendly)
    plt.rcParams.update({
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    })

    models = ["2B", "9B"]
    methods = ["Baseline", "Steering (Q)", "Steering (C)"]
    x = np.arange(len(models))
    width = 0.25

    def get_vals(metric: str, method: str) -> List[float]:
        out = []
        for m in models:
            v = df[(df["Model"] == m) & (df["Method"] == method)][metric].values
            out.append(float(v[0]) if len(v) else float("nan"))
        return out

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 2.9),  
        sharex=True,
        sharey=True
    )

    panels = [
        ("clarif_rate", "Clarification rate"),
        ("acc", "Task success"),
    ]

    legend_handles = None
    legend_labels = None

    for ax, (metric, title) in zip(axes, panels):
        b_baseline = ax.bar(x - width, get_vals(metric, "Baseline"), width,
                            label="Baseline", color=PALETTE["Baseline"])
        b_q = ax.bar(x, get_vals(metric, "Steering (Q)"), width,
                     label="Steering (Q)", color=PALETTE["Steering (Q)"])
        b_c = ax.bar(x + width, get_vals(metric, "Steering (C)"), width,
                     label="Steering (C)", color=PALETTE["Steering (C)"])

        ax.set_title(title, fontsize=FONT_TITLE, pad=TITLE_PAD)
        ax.set_xticks(x, models)
        ax.set_ylim(0, 1.0)
        nice_axes(ax)
        annotate(ax, b_baseline)
        annotate(ax, b_q)
        annotate(ax, b_c)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Shared legend once (top center) and shared xlabel once
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995)  # inside, very top
    )
    fig.supxlabel("Model size", y=SUPXLABEL_Y)

    fig.subplots_adjust(
        left=0.08, right=0.995,
        bottom=0.18, top=0.86,  
        wspace=0.12
    )

    fig.tight_layout(rect=(0, 0.0, 1, 0.90))

    out_base = args.out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"\nWrote combined figure: {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")
    print(f"Wrote CSV (for provenance): {args.out_dir/'ambik_main_bars.csv'}")


if __name__ == "__main__":
    main()
