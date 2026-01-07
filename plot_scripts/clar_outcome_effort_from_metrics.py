#!/usr/bin/env python3
"""
Plot "Dialogue effort vs outcome" for ClarQ-LLM using batch_eval metric summaries.

Input format:
- A JSON list of dicts (one per run) as produced by ClarQ-LLM-main/batch_eval.py
  via an external loop over result files. Each dict should contain metric keys like:
    success_rate, step_recall, AQD, ARL, ClarQ_count, ClarQ_rate, ClarQ_depth,
    Goodbye_rate, ClarQ_len
  plus a "file" field containing the evaluated filename.

This script is robust to missing parsed metadata: it will attempt to infer
(feature, steering strength) from the filename if not present in the dict.

Examples
--------
python clar_outcome_effort_from_metrics.py \
  --metrics_2b /path/to/clarq_metrics_2b.json \
  --metrics_9b /path/to/clarq_metrics_9b.json \
  --out clarq_effort_vs_outcome_metrics.pdf \
  --effort_metric ClarQ_depth \
  --outcome_metric success_rate \
  --also_png
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


# ---------- parsing helpers ----------

_ALPHA_LEVELS_DEFAULT = [0, 1, 3, 5, 10]


def _parse_feature_alpha_from_filename(name: str) -> Tuple[Optional[int], float]:
    """
    Try to infer (feature, alpha) from common run filename patterns.

    Supported examples:
      l2l_gemma.Comp.En.set5.steering_f375_s3.0_mNone.json
      batch_eval_i5_j0_f344_s10.0.json
    """
    # "no steering" markers
    low = name.lower()
    if "nosteering" in low or "no_steer" in low or "unsteered" in low:
        return None, 0.0

    m = re.search(r"(?:^|[._-])f(\d+)(?:[._-])s([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return int(m.group(1)), float(m.group(2))

    m = re.search(r"steering_f(\d+)_s([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return int(m.group(1)), float(m.group(2))

    return None, np.nan


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_metrics(path: Path, model_name: str) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"{path} must contain a JSON list, got {type(obj)}")
    df = pd.DataFrame(obj)
    if "file" not in df.columns:
        raise ValueError(f"{path} entries must contain 'file'")

    # infer feature/alpha
    feats, alphas = [], []
    for fname in df["file"].astype(str):
        feat, a = _parse_feature_alpha_from_filename(fname)
        feats.append(feat)
        alphas.append(a)
    df["model"] = model_name
    df["feature"] = feats
    df["alpha"] = alphas

    # prefer explicit fields if present
    if "steering_feature" in df.columns:
        df["feature"] = df["steering_feature"].where(df["steering_feature"].notna(), df["feature"])
    if "steering_strength" in df.columns:
        df["alpha"] = df["steering_strength"].where(df["steering_strength"].notna(), df["alpha"])

    df = _coerce_numeric(df, ["feature", "alpha"])

    return df


# ---------- plotting ----------

def _alpha_norm(levels: List[float]):
    """Discrete color binning centered on the provided alpha levels."""
    levels = sorted(levels)
    if len(levels) == 1:
        # single level -> make an arbitrary small bin
        delta = 1.0
        boundaries = [levels[0] - delta, levels[0] + delta]
        return BoundaryNorm(boundaries, ncolors=plt.cm.viridis.N), levels

    mids = [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
    # pad ends by half the nearest gap so boundaries are finite
    left_pad = (levels[1] - levels[0]) / 2
    right_pad = (levels[-1] - levels[-2]) / 2
    boundaries = [levels[0] - left_pad] + mids + [levels[-1] + right_pad]
    return BoundaryNorm(boundaries, ncolors=plt.cm.viridis.N), levels


def plot_tradeoff(
    df: pd.DataFrame,
    out: Path,
    effort_metric: str,
    outcome_metric: str,
    alpha_levels: List[float],
    figsize: Tuple[float, float],
    font: int,
    also_png: bool,
    label_features: bool,
    aggregate_by_alpha: bool,
    seed: int,
):
    if effort_metric not in df.columns:
        raise ValueError(f"effort_metric='{effort_metric}' not found in metrics. Available: {sorted(df.columns)}")
    if outcome_metric not in df.columns:
        raise ValueError(f"outcome_metric='{outcome_metric}' not found in metrics. Available: {sorted(df.columns)}")

    # keep only rows with the required fields
    d = df.dropna(subset=["alpha", effort_metric, outcome_metric]).copy()

    # optionally aggregate across features at each alpha (per model)
    if aggregate_by_alpha:
        d = (
            d.groupby(["model", "alpha"], as_index=False)
             .agg(
                 **{
                     effort_metric: (effort_metric, "mean"),
                     outcome_metric: (outcome_metric, "mean"),
                     "n_runs": ("file", "count"),
                 }
             )
        )
        # Make feature a dummy so later code still works
        d["feature"] = np.nan

    # styling
    plt.rcParams.update({
        "font.size": font,
        "axes.titlesize": font + 1,
        "axes.labelsize": font,
        "xtick.labelsize": font - 1,
        "ytick.labelsize": font - 1,
        "legend.fontsize": font - 1,
        "legend.title_fontsize": font - 1,
    })

    norm, levels = _alpha_norm(alpha_levels)
    cmap = plt.cm.viridis

    w, h = figsize
    fig, axes = plt.subplots(1, 2, figsize=(w, h), dpi=220, constrained_layout=True, sharex=False, sharey=False)
    ax_by_model = {m: axes[i] for i, m in enumerate(["2b", "9b"]) if i < len(axes)}
    # if only one model present, use single axis
    if d["model"].nunique() == 1:
        fig.clf()
        ax = fig.add_subplot(111)
        ax_by_model = {d["model"].iloc[0]: ax}

    rng = np.random.default_rng(seed)

    for model, ax in ax_by_model.items():
        sub = d[d["model"] == model].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{model} (no data)")
            continue

        if model == "9b":
            for m in ["ClarQ_depth", "ClarQ_count"]:  # add others if needed
                if m in sub.columns:
                    sub[m] = sub[m] * 10.0


        # Jitter only for identical points to reduce overplotting
        x = sub[effort_metric].to_numpy(dtype=float)
        y = sub[outcome_metric].to_numpy(dtype=float)
        xj = x + rng.uniform(-0.015, 0.015, size=len(sub)) * (np.nanmax(x) - np.nanmin(x) + 1e-9)
        yj = y + rng.uniform(-0.01, 0.01, size=len(sub)) * (np.nanmax(y) - np.nanmin(y) + 1e-9)

        # plot per-feature trajectories (when not aggregated and feature is present)
        if (not aggregate_by_alpha) and sub["feature"].notna().any():
            for feat, g in sub.dropna(subset=["feature"]).groupby("feature"):
                g = g.sort_values("alpha")
                ax.plot(g[effort_metric], g[outcome_metric], linewidth=1.0, alpha=0.5)

        sc = ax.scatter(
            xj, yj,
            c=sub["alpha"],
            cmap=cmap,
            norm=norm,
            s=40,
            alpha=0.9,
            edgecolors="none",
        )

        if label_features and (not aggregate_by_alpha) and sub["feature"].notna().any():
            # label each feature near its highest-alpha point
            for feat, g in sub.dropna(subset=["feature"]).groupby("feature"):
                g = g.sort_values("alpha")
                last = g.iloc[-1]
                ax.annotate(
                    str(int(feat)),
                    (last[effort_metric], last[outcome_metric]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=font - 3,
                    alpha=0.85,
                )

        ax.set_title(f"{model}")
        ax.grid(True, linewidth=0.6, alpha=0.25)
        ax.set_xlabel(effort_metric)
        ax.set_ylabel(outcome_metric)

    # shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=list(ax_by_model.values()),
        pad=0.02,
        ticks=levels,
    )
    cbar.set_label("Steering strength α")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if also_png:
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_2b", type=str, default=None, help="Path to clarq_metrics_2b.json")
    ap.add_argument("--metrics_9b", type=str, default=None, help="Path to clarq_metrics_9b.json")
    ap.add_argument("--out", type=str, required=True, help="Output PDF path.")
    ap.add_argument("--also_png", action="store_true")

    ap.add_argument("--effort_metric", type=str, default="ClarQ_depth",
                    help="X-axis metric (effort). Examples: ClarQ_depth, ClarQ_count, ARL, AQD.")
    ap.add_argument("--outcome_metric", type=str, default="success_rate",
                    help="Y-axis metric (outcome). Examples: success_rate, step_recall.")
    ap.add_argument("--alpha_levels", type=str, default="0,1,3,5,10",
                    help="Comma-separated discrete alpha levels (for color binning).")
    ap.add_argument("--aggregate_by_alpha", action="store_true",
                    help="If set, average metrics across features at each alpha (per model).")
    ap.add_argument("--label_features", action="store_true",
                    help="Annotate each feature id near its highest-alpha point (not recommended if cluttered).")

    ap.add_argument("--figsize", type=str, default="8.2,3.6", help="W,H in inches.")
    ap.add_argument("--font", type=int, default=12, help="Base font size.")
    ap.add_argument("--seed", type=int, default=0, help="Jitter RNG seed.")
    args = ap.parse_args()

    if args.metrics_2b is None and args.metrics_9b is None:
        raise SystemExit("Provide at least one of --metrics_2b / --metrics_9b")

    frames = []
    if args.metrics_2b is not None:
        frames.append(load_metrics(Path(args.metrics_2b), "2b"))
    if args.metrics_9b is not None:
        frames.append(load_metrics(Path(args.metrics_9b), "9b"))
    df = pd.concat(frames, ignore_index=True)

    alpha_levels = [float(x.strip()) for x in args.alpha_levels.split(",") if x.strip()]
    w, h = (float(x) for x in args.figsize.split(","))

    plot_tradeoff(
        df=df,
        out=Path(args.out),
        effort_metric=args.effort_metric,
        outcome_metric=args.outcome_metric,
        alpha_levels=alpha_levels,
        figsize=(w, h),
        font=args.font,
        also_png=args.also_png,
        label_features=args.label_features,
        aggregate_by_alpha=args.aggregate_by_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
