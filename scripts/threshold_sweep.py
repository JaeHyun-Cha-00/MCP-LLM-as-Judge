#!/usr/bin/env python
"""Plot realized MAE / Spearman ρ as a function of the fallback threshold.

For each (dataset, metric, k), sweeps threshold T over a fine grid, computes
the per-rubric routing operating point at every T, and plots realized metric
on the left y-axis and fraction of rubrics escalated on the right y-axis.

Saves PDF + PNG (one figure per (dataset, metric) pair) to scripts/figures/.

Run from the project root:

    python scripts/threshold_sweep.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse all the data plumbing from pareto_figures so the policy stays in sync.
from pareto_figures import (
    DATASETS,
    PROJECT_ROOT,
    STAT_STYLES,
    bootstrap_calibration,
    compute_cat_corr,
    load_dataset,
    operating_point_at_threshold,
)


METRIC_SPEC = {
    "mae": {
        "score_col": "ucb_mae",
        "higher_is_better": False,
        "is_error": True,
        "ylabel": "Realized MAE  (lower = better)",
        "title_metric": "MAE",
    },
    "rmse": {
        "score_col": "ucb_rmse",
        "higher_is_better": False,
        "is_error": True,
        "ylabel": "Realized RMSE  (lower = better)",
        "title_metric": "RMSE",
    },
    "rho": {
        "score_col": "lcb_rho",
        "higher_is_better": True,
        "is_error": False,
        "ylabel": "Realized Spearman ρ  (higher = better)",
        "title_metric": "Spearman ρ",
    },
    "kendall": {
        "score_col": "lcb_kendall",
        "higher_is_better": True,
        "is_error": False,
        "ylabel": "Realized Kendall τ  (higher = better)",
        "title_metric": "Kendall τ",
    },
}


def threshold_sweep(score_table, value_lookup, categories, n_per_cat,
                    higher_is_better, thresholds):
    """Run operating_point_at_threshold across a vector of thresholds.

    Returns a DataFrame with columns [threshold, metric, n_escalated, calls].
    """
    rows = []
    for thr in thresholds:
        op = operating_point_at_threshold(
            score_table, value_lookup, categories,
            0, n_per_cat, higher_is_better, thr,
        )
        rows.append({
            "threshold": float(thr),
            "metric": op["metric"],
            "n_escalated": op["n_escalated"],
            "calls": op["calls"],
        })
    return pd.DataFrame(rows)


def make_figure(cfg, metric, k_values, alpha, seed, n_points):
    if metric not in METRIC_SPEC:
        raise ValueError(f"Unknown metric: {metric}")
    spec = METRIC_SPEC[metric]
    higher_is_better = spec["higher_is_better"]
    score_col = spec["score_col"]
    ylabel = spec["ylabel"]

    pairs, errors, evaluator_names = load_dataset(cfg)
    n_total = len(cfg.categories)
    per_q_min = errors.groupby(cfg.id_keys + ["category"])["abs_err"].min().reset_index()
    n_per_cat = len(per_q_min) // n_total

    if metric == "mae":
        cat_value = (errors.groupby(["evaluator", "category"])["abs_err"].mean()
                     .unstack("evaluator")
                     .reindex(index=cfg.categories, columns=evaluator_names))
    elif metric == "rmse":
        cat_value = (errors.groupby(["evaluator", "category"])["sq_err"].mean()
                     .pow(0.5)
                     .unstack("evaluator")
                     .reindex(index=cfg.categories, columns=evaluator_names))
    elif metric == "rho":
        cat_value = compute_cat_corr(pairs, cfg.categories, evaluator_names,
                                     method="spearman")
    else:  # kendall
        cat_value = compute_cat_corr(pairs, cfg.categories, evaluator_names,
                                     method="kendall")

    if spec["is_error"]:
        # Sweep thresholds from "escalate all" (0) to "escalate none"
        # (a comfortable margin past the worst calibrated value we'll see).
        thresholds = np.linspace(0.0, cfg.score_range, n_points)
        xlabel = (f"{spec['title_metric']} fallback threshold T  "
                  f"(score range = {cfg.score_range:g})")
    else:
        # Correlation ∈ [-1, 1]; "escalate none" at T=-1, "escalate all" at T=1.
        thresholds = np.linspace(-1.0, 1.0, n_points)
        sym = "ρ" if metric == "rho" else "τ"
        xlabel = f"{sym} fallback threshold T"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax2 = ax.twinx()
    ax2.grid(False)

    for k in k_values:
        cal = bootstrap_calibration(pairs, k=k, alpha=alpha, seed=seed)
        score_table = (cal.pivot_table(index="category", columns="evaluator",
                                       values=score_col)
                       .reindex(index=cfg.categories, columns=evaluator_names))

        sweep = threshold_sweep(score_table, cat_value, cfg.categories,
                                n_per_cat, higher_is_better, thresholds)

        color = STAT_STYLES.get(k, {"color": "#555555"})["color"]
        ax.plot(sweep["threshold"], sweep["metric"], color=color, lw=2.0,
                label=f"k={k}  realized {spec['title_metric']}", zorder=4)
        ax2.plot(sweep["threshold"], sweep["n_escalated"] / n_total,
                 color=color, lw=1.0, ls="--", alpha=0.55,
                 label=f"k={k}  fraction escalated", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax2.set_ylabel(f"Fraction of {n_total} rubrics escalated  (dashed)")
    ax2.set_ylim(-0.02, 1.02)

    if higher_is_better:
        ax.set_ylim(top=1.02)
    else:
        ax.set_ylim(bottom=0.0)

    title = f"Threshold sweep — {cfg.label}  ({spec['title_metric']})"
    ax.set_title(title, fontweight="bold")

    # Combine legends from both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, framealpha=0.95)

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "scripts" / "figures",
                   help="Where to save PDF/PNG files (default: scripts/figures)")
    p.add_argument("--metrics", nargs="+", default=["mae", "rmse", "rho", "kendall"],
                   choices=["mae", "rmse", "rho", "kendall"],
                   help="Which metric(s) to plot (default: all four)")
    p.add_argument("--k-values", nargs="+", type=int, default=[5],
                   help="Bootstrap k values to overlay (default: 5)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Confidence level for UCB/LCB (default: 0.05)")
    p.add_argument("--seed", type=int, default=0,
                   help="Bootstrap seed (default: 0)")
    p.add_argument("--n-points", type=int, default=200,
                   help="Number of thresholds in the sweep (default: 200)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        for metric in args.metrics:
            print(f"[{cfg.name} / {metric}] sweeping ...")
            fig = make_figure(cfg, metric=metric, k_values=args.k_values,
                              alpha=args.alpha, seed=args.seed,
                              n_points=args.n_points)
            base = args.output_dir / f"threshold_sweep_{cfg.name}_{metric}"
            fig.savefig(f"{base}.pdf", bbox_inches="tight")
            fig.savefig(f"{base}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  saved {base.name}.pdf / .png")


if __name__ == "__main__":
    main()
