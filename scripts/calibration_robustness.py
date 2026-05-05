#!/usr/bin/env python
"""Calibration-robustness figure: variance across bootstrap seeds at fixed m.

For each (dataset, metric, k) we run `bootstrap_calibration` `--n-seeds` times
with m = ⌊√n⌋ held fixed. For every (rubric × evaluator) pair we compute the
std of the calibrated routing-score (UCB-MAE / UCB-RMSE / LCB-ρ / LCB-τ) across
seeds, then plot the distribution of those stds as a violin per k.

Lower std = more stable calibration. Larger k should shrink the spread.

Saves PDF + PNG (one figure per (dataset, metric) pair) to scripts/figures/.

Run from the project root:

    python scripts/calibration_robustness.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pareto_figures import (
    DATASETS,
    PROJECT_ROOT,
    bootstrap_calibration,
    compute_cat_value,
    load_dataset,
    metric_spec,
    operating_point_at_threshold,
)


# Routing-score column per metric — same one the policy thresholds against in
# pareto_figures / threshold_sweep, so the noise we measure here is the noise
# that actually affects routing decisions.
SCORE_COL = {
    "mae": "ucb_mae",
    "rmse": "ucb_rmse",
    "rho": "lcb_rho",
    "kendall": "lcb_kendall",
}

METRIC_LABEL = {
    # "mae": "UCB MAE",
    # "rmse": "UCB RMSE",
    # "rho": "LCB Spearman ρ",
    # "kendall": "LCB Kendall τ",
    "mae": "MAE",
    "rmse": "RMSE",
    "rho": "Spearman ρ",
    "kendall": "Kendall τ",
}

# k → color. Reds for tiny k (high noise), cooler tones for larger k.
K_PALETTE = {
    1:  "#E63946",
    2:  "#F4A261",
    5:  "#2E86AB",
    10: "#3B1F2B",
    50: "#A23B72",
}


def collect_calibrations(pairs, k, n_seeds, alpha):
    """Run bootstrap_calibration for seeds 0..n_seeds-1 and return the list."""
    return [bootstrap_calibration(pairs, k=k, alpha=alpha, seed=s)
            for s in range(n_seeds)]


def per_call_realized_metric(cal_list, cat_value, categories, evaluator_names,
                             score_col, higher_is_better, n_per_cat):
    """Realized macro-quality under the no-fallback policy, one value per call.

    Mirrors the Pareto's leftmost MoJ-stat marker: per call, build the calibrated
    `score_col` table, pick the best surrogate per rubric, look up its full-data
    quality from `cat_value`, and macro-average across rubrics. Across-seed
    spread of the result isolates seed noise in the chosen-surrogate quality
    that the routing policy actually delivers.
    """
    no_fallback = float("-inf") if higher_is_better else float("inf")
    means = []
    for cal in cal_list:
        score_table = (cal.pivot_table(index="category", columns="evaluator",
                                       values=score_col)
                       .reindex(index=categories, columns=evaluator_names))
        op = operating_point_at_threshold(
            score_table, cat_value, categories,
            0, n_per_cat, higher_is_better, no_fallback,
        )
        means.append(op["metric"])
    return means


def make_figure(cfg, metric, cal_results, evaluator_names, k_values, n_seeds,
                cat_value, n_per_cat):
    spec = metric_spec(metric)
    score_col = spec["score_col"]
    higher_is_better = spec["higher_is_better"]

    rows = []
    for k in k_values:
        means = per_call_realized_metric(
            cal_results[k], cat_value, cfg.categories, evaluator_names,
            score_col, higher_is_better, n_per_cat,
        )
        for s, v in enumerate(means):
            rows.append({"k": k, "seed": s, "score": v})
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4.5, 4))

    palette = [K_PALETTE.get(k, "#888888") for k in k_values]
    sns.violinplot(
        data=df, x="k", y="score", order=k_values,
        inner="quartile", cut=0, 
        density_norm="width",
        palette=palette, ax=ax, linewidth=1.0,
    )
    sns.stripplot(
        data=df, x="k", y="score", order=k_values,
        size=1.8, color="#222222", alpha=0.25, jitter=0.18, ax=ax,
    )

    for i, k in enumerate(k_values):
        sub = df[df["k"] == k]["score"]
        if len(sub) == 0:
            continue
        med = float(sub.median())
        ax.text(i, med, f"  med={med:.3f}", va="center", ha="left",
                fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.7))

    ax.set_xlabel("k samples per ⌊√n⌋")
    ax.set_ylabel(f"{METRIC_LABEL[metric]} under no-fallback\n"
                  f"(one value per calibration; {n_seeds} seeds per k)")
    ax.set_title(f"{cfg.label}"
                 f"(m = ⌊√n⌋ fixed)", fontweight="bold")
    # if higher_is_better:
    #     ax.set_ylim(top=1.05)
    # else:
    #     ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.grid(False, axis="x")
    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "figures",
                   help="Where to save PDF/PNG files (default: figures/)")
    p.add_argument("--metrics", nargs="+", default=["mae", "rmse", "rho", "kendall"],
                   choices=["mae", "rmse", "rho", "kendall"],
                   help="Which metric(s) to plot (default: all four)")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5, 10],
                   help="Bootstrap k values to compare (default: 1 2 5 10)")
    p.add_argument("--n-seeds", type=int, default=50,
                   help="Number of bootstrap seeds per k (default: 50)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Confidence level for UCB/LCB (default: 0.05)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        pairs, errors, evaluator_names = load_dataset(cfg)
        per_q_min = (errors.groupby(cfg.id_keys + ["category"])["abs_err"]
                     .min().reset_index())
        n_per_cat = len(per_q_min) // len(cfg.categories)
        print(f"[{cfg.name}] running bootstraps  k={args.k_values}  "
              f"n_seeds={args.n_seeds} ...")
        cal_results = {
            k: collect_calibrations(pairs, k, args.n_seeds, args.alpha)
            for k in args.k_values
        }

        for metric in args.metrics:
            print(f"  [{metric}] generating ...")
            cat_value = compute_cat_value(pairs, errors, metric,
                                          cfg.categories, evaluator_names)
            fig = make_figure(cfg, metric, cal_results, evaluator_names,
                              args.k_values, args.n_seeds,
                              cat_value=cat_value, n_per_cat=n_per_cat)
            base = args.output_dir / f"calibration_robustness_{cfg.name}_{metric}"
            fig.savefig(f"{base}.pdf", bbox_inches="tight")
            fig.savefig(f"{base}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"    saved {base.name}.pdf / .png")


if __name__ == "__main__":
    main()
