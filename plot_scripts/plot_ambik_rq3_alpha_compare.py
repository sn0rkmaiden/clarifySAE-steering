#!/usr/bin/env python3
import argparse, json, zipfile, math
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_json_from_zip(z: zipfile.ZipFile, member: str) -> dict:
    with z.open(member) as f:
        return json.loads(f.read().decode("utf-8"))

def iter_jsons(results: Path):
    if results.is_file() and results.suffix.lower() == ".zip":
        with zipfile.ZipFile(results, "r") as z:
            for m in z.namelist():
                if m.lower().endswith(".json"):
                    try:
                        yield m, load_json_from_zip(z, m)
                    except Exception:
                        continue
    else:
        for p in results.rglob("*.json"):
            try:
                yield str(p), json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue


def nanmean(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs2)) if xs2 else float("nan")

def nansem(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs2) <= 1:
        return float("nan")
    return float(np.std(xs2, ddof=1) / math.sqrt(len(xs2)))

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
    # 2B naming
    if "results_2b_question" in s:
        return "Q"
    if "results_2b_clar" in s:
        return "C"
    # 9B naming
    if "question_vocab" in s:
        return "Q"
    if "clar_vocab" in s or "clar_vocab2" in s:
        return "C"
    return None

def get_alpha(obj: dict) -> Optional[float]:
    cfg = obj.get("run_info", {}).get("steering_cfg", {})
    a = cfg.get("strength", None)
    return float(a) if a is not None else None

def get_sae_id(obj: dict) -> Optional[str]:
    return obj.get("run_info", {}).get("steering_cfg", {}).get("sae_id", None)

def compute_ambik_metrics(obj: dict) -> Dict[str, float]:
    ex = obj.get("examples", [])
    clar = [1 if e.get("num_questions", 0) > 0 else 0 for e in ex]
    acc = [1 if e.get("resolved_proxy", False) else 0 for e in ex]
    return {
        "n_examples": len(ex),
        "clarif_rate": nanmean(clar),
        "acc": nanmean(acc),
    }


def pick_baseline(items: List[Tuple[str, dict]], model: str) -> Tuple[str, dict]:
    """
    Prefer a baseline file that is clearly within the right model subtree.
    """
    # Strong heuristics based on your zip structure
    if model == "2B":
        for name, obj in items:
            if not is_steered(obj) and "2bmodel" in name.lower() and "baseline" in name.lower():
                return name, obj
    if model == "9B":
        # prefer layer20 question baseline (matches the 9B Q runs)
        for name, obj in items:
            if not is_steered(obj) and "9bmodel" in name.lower() and "question_vocab" in name.lower() and "baseline" in name.lower():
                return name, obj
        for name, obj in items:
            if not is_steered(obj) and "9bmodel" in name.lower() and "baseline" in name.lower():
                return name, obj

    # fallback: any unsteered with correct model_name
    target = "gemma-2b-it" if model == "2B" else "gemma-2-9b-it"
    for name, obj in items:
        if not is_steered(obj) and model_name(obj) == target:
            return name, obj

    raise RuntimeError(f"Could not locate baseline for {model}.")


plt.rcParams.update({
    "font.family": "serif",
    # Slightly larger typography for paper readability
    "font.size": 12,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# Styling knobs (paper-friendly defaults)
PLOT_LW = 2.2
BASELINE_LW = 1.8
GRID_LW = 0.9

# Layout knobs
LEGEND_Y = 0     # raise legend closer to x-axis labels (was -0.02)
TIGHT_BOTTOM = 0.10   # allow subplots to extend lower (was 0.14)
XLABEL_PAD = 2        # reduce gap between xlabel and axes

COLORS = {"Q": "#FF9013", "C": "#D73535"}  # muted blue/orange
LABELS = {"Q": "Q vocabulary", "C": "C vocabulary"}

def nice_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=GRID_LW)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="results_without_rand.zip (recommended) or a directory.")
    ap.add_argument("--sae_id_2b", type=str, default="blocks.12.hook_resid_post",
                    help="Filter 2B runs to this SAE id (default matches your data).")
    ap.add_argument("--sae_id_9b", type=str, default="blocks.20.hook_resid_post",
                    help="Filter 9B runs to this SAE id to match Q vs C (default blocks.20).")
    ap.add_argument("--out_dir", type=Path, default=Path("plots_out"))
    ap.add_argument("--out_name", type=str, default="ambik_rq3_alpha_compare")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_jsons(args.results))

    # Baselines
    b2_name, b2_obj = pick_baseline(items, "2B")
    b9_name, b9_obj = pick_baseline(items, "9B")
    b2 = compute_ambik_metrics(b2_obj)
    b9 = compute_ambik_metrics(b9_obj)

    # Collect steered runs with SAE-id filtering for a fair vocab comparison
    rows = []
    for name, obj in items:
        if not is_steered(obj):
            continue

        mk = model_key(obj, name)
        vocab = infer_vocab(name)
        alpha = get_alpha(obj)
        sae_id = get_sae_id(obj)

        if mk is None or vocab is None or alpha is None:
            continue

        if mk == "2B" and sae_id != args.sae_id_2b:
            continue

        layer_tag = None
        if mk == "9B":
            if sae_id == "blocks.20.hook_resid_post":
                layer_tag = "L20"
            elif sae_id == "blocks.31.hook_resid_post":
                layer_tag = "L31"
            else:
                continue  # ignore other layers if present
        else:
            layer_tag = "L12"  # 2B fixed layer


        m = compute_ambik_metrics(obj)
        rows.append({
            "path": name,
            "model": mk,
            "vocab": vocab,
            "layer": layer_tag,
            "alpha": alpha,
            "clarif_rate": m["clarif_rate"],
            "acc": m["acc"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No runs after filtering. Check --sae_id_2b/--sae_id_9b against your zip contents.")
    df = df[~((df["model"] == "9B") & (df["layer"] == "L20") & (df["alpha"] == 30))]
    # Aggregate across runs at same (model,vocab,alpha)
    agg = (
        df.groupby(["model", "vocab", "layer", "alpha"])
          .agg(
              mean_clarif=("clarif_rate", "mean"),
              sem_clarif=("clarif_rate", nansem),
              mean_acc=("acc", "mean"),
              sem_acc=("acc", nansem),
              n_runs=("acc", "count"),
          )
          .reset_index()
          .sort_values(["model", "vocab", "alpha"])
    )

    out_csv = args.out_dir / f"{args.out_name}.csv"
    agg.to_csv(out_csv, index=False)

    # Print coverage (so you can see what alphas exist)
    print("\n=== Using baselines ===")
    print(f"2B baseline: {b2_name}  (clar={b2['clarif_rate']:.3f}, acc={b2['acc']:.3f})")
    print(f"9B baseline: {b9_name}  (clar={b9['clarif_rate']:.3f}, acc={b9['acc']:.3f})")
    print("\n=== SAE-id filters ===")
    print(f"2B: {args.sae_id_2b}")
    print(f"9B: {args.sae_id_9b}")

    print("\n=== Alpha coverage after filtering (counts = n_runs) ===")
    for mk in ["2B", "9B"]:
        for vocab in ["Q", "C"]:
            sub = agg[(agg.model == mk) & (agg.vocab == vocab)]
            if sub.empty:
                print(f"{mk} {vocab}: <none>")
            else:
                pairs = [(float(a), int(n)) for a, n in zip(sub.alpha, sub.n_runs)]
                print(f"{mk} {vocab}: {pairs}")

    COLORS = {
        ("Q", "L20"): "#FF9013",   
        ("C", "L20"): "#0046FF",   
        ("C", "L31"): "#73C8D2",  
        ("Q", "L12"): "#0046FF",
        ("C", "L12"): "#FF9013",
    }
    LINESTYLES = {
        ("Q", "L20"): "-",
        ("C", "L20"): "-",
        ("C", "L31"): "--",       
        ("Q", "L12"): "-",
        ("C", "L12"): "-",
    }
    LABELS = {
        ("Q", "L20"): "Q vocabulary (L20)",
        ("C", "L20"): "C vocabulary (L20)",
        ("C", "L31"): "C vocabulary (L31)",
        ("Q", "L12"): "Q vocabulary (L12)",
        ("C", "L12"): "C vocabulary (L12)",

    }


    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.2), sharex=False)

    # y-lims: clarification fixed, acc tightened globally
    axes[0, 0].set_title("Clarification rate")
    axes[0, 1].set_title("Task success")

    # compute acc y-lims from all plotted points (plus baseline)
    acc_vals = agg["mean_acc"].dropna().tolist() + [b2["acc"], b9["acc"]]
    lo = max(0.0, min(acc_vals) - 0.02) if acc_vals else 0.0
    hi = min(1.0, max(acc_vals) + 0.06) if acc_vals else 1.0

    for r, mk in enumerate(["2B", "9B"]):
        # left panel: clarif
        axL = axes[r, 0]
        nice_axes(axL)
        axL.set_ylim(0, 0.75)
        axL.set_yticks([0.0, 0.25, 0.5, 0.75])
        axL.set_ylabel(mk)

        # right panel: acc
        axR = axes[r, 1]
        nice_axes(axR)
        axR.set_ylim(lo, hi)

        baseline = b2 if mk == "2B" else b9
        axL.axhline(baseline["clarif_rate"], color="#CF0F0F", linestyle="--", linewidth=BASELINE_LW)
        axR.axhline(baseline["acc"],         color="#CF0F0F", linestyle="--", linewidth=BASELINE_LW)

        if mk == "2B":
            curve_keys = [("Q","L12"), ("C","L12")]  # if you actually have both; otherwise keep as before
        else:
            curve_keys = [("Q","L20"), ("C","L20"), ("C","L31")]

        for vocab, layer in curve_keys:
            sub = agg[(agg.model == mk) & (agg.vocab == vocab) & (agg.layer == layer)].sort_values("alpha")
            if sub.empty:
                continue
            x = sub.alpha.values

            axL.errorbar(
                x, sub.mean_clarif.values, yerr=sub.sem_clarif.values,
                color=COLORS[(vocab, layer)],
                linestyle=LINESTYLES[(vocab, layer)],
                marker="o", linewidth=PLOT_LW, markersize=5, capsize=2,
                label=None
            )
            axR.errorbar(
                x, sub.mean_acc.values, yerr=sub.sem_acc.values,
                color=COLORS[(vocab, layer)],
                linestyle=LINESTYLES[(vocab, layer)],
                marker="o", linewidth=PLOT_LW, markersize=5, capsize=2
            )


    axes[1, 0].set_xlabel(r"Steering strength $\alpha$", labelpad=XLABEL_PAD)
    axes[1, 1].set_xlabel(r"Steering strength $\alpha$", labelpad=XLABEL_PAD)

    baseline_handle = Line2D([0], [0], color="#CF0F0F", linestyle="--", linewidth=BASELINE_LW, label="Baseline")

    q_handle = Line2D([0], [0],
                    color=COLORS[("Q", "L12")], linestyle="-", marker="o",
                    linewidth=PLOT_LW, markersize=5,
                    label="Q vocabulary (2B: L12, 9B: L20)")

    c20_handle = Line2D([0], [0],
                        color=COLORS[("C", "L12")], linestyle="-", marker="o",
                        linewidth=PLOT_LW, markersize=5,
                        label="C vocabulary (2B: L12, 9B: L20)")

    c31_handle = Line2D([0], [0],
                        color=COLORS[("C", "L31")], linestyle="--", marker="o",
                        linewidth=PLOT_LW, markersize=5,
                        label="C vocabulary (9B: L31)")

    fig.legend(
        [q_handle, c20_handle, c31_handle, baseline_handle],
        ["Q vocabulary (2B: L12, 9B: L20)",
        "C vocabulary (2B: L12, 9B: L20)",
        "C vocabulary (9B: L31)",
        "Baseline"],
        loc="lower center",
        ncol=2,          # 2 columns keeps it readable
        frameon=False,
        bbox_to_anchor=(0.5, LEGEND_Y)
    )


    # Leave room at the bottom for the legend
    fig.tight_layout(rect=(0, TIGHT_BOTTOM, 1, 1))

    out_base = args.out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_base.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"\nWrote: {out_base.with_suffix('.pdf')}")
    print(f"Wrote: {out_base.with_suffix('.png')}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
