#!/usr/bin/env python
"""Generate # frontier-calls vs. quality Pareto figures for both benchmarks.

Reads validation pairs from creative_writing/ and story_writing_benchmark/,
runs bootstrap calibration, builds MoJ-stat / oracle Pareto curves, and saves
PDF + PNG (one figure per (dataset, metric) pair) to scripts/figures/.

Run from the project root:

    python scripts/pareto_figures.py
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Dataset configs
# =============================================================================

@dataclass
class DatasetConfig:
    name: str
    label: str
    results_dir: Path
    baseline_path: Optional[Path]
    categories: list
    id_keys: list
    pretty_name: Callable[[str], str]
    glob_pattern: str
    score_range: float  # max - min possible per rubric; used to normalize MAE thresholds
    ref_col_for: Optional[Callable[[str], str]] = None  # only when refs live in result rows


def _sw_pretty(stem: str) -> str:
    s = stem.replace("_swb_result", "").replace("_result", "")
    s = s.replace("-Instruct-2507", "").replace("-Instruct", "")
    s = s.replace("_it", "-it").replace("-it", "")
    return s


def _cw_pretty(stem: str) -> str:
    s = stem.replace("_result", "")
    s = s.replace("-Instruct", "").replace("-it", "").replace("-2507", "")
    return s


SW_CATEGORIES = [
    "Grammar, Spelling, and Punctuation Quality",
    "Clarity and Understandability",
    "Logical Connection Between Events and Ideas",
    "Scene Construction and Purpose",
    "Internal Consistency",
    "Character Consistency",
    "Character Motivation and Actions",
    "Sentence Pattern Variety",
    "Avoidance of Clichés and Overused Phrases",
    "Natural Dialogue",
    "Avoidance of Predictable Narrative Tropes",
    "Character Depth and Dimensionality",
    "Realistic Character Interactions",
    "Ability to Hold Reader Interest",
    "Satisfying Plot Resolution",
]
SW_REF = {c: f"ref_q{i+1}" for i, c in enumerate(SW_CATEGORIES)}


CW_CATEGORIES = [
    # positive (higher = better in raw scoring; for MAE/rho both are valid)
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice / Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
    "Overall Impression",
    # negative (lower = better in raw scoring)
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
]


DATASETS = [
    DatasetConfig(
        name="story_writing_benchmark",
        label="Story Writing Dataset",
        results_dir=PROJECT_ROOT / "story_writing_benchmark" / "dataset" / "results",
        baseline_path=None,
        categories=SW_CATEGORIES,
        id_keys=["index", "prompt_id", "model"],
        pretty_name=_sw_pretty,
        glob_pattern="*.csv",
        score_range=5.0,
        ref_col_for=lambda cat: SW_REF[cat],
    ),
    DatasetConfig(
        name="creative_writing",
        label="Creative Writing Dataset",
        results_dir=PROJECT_ROOT / "creative_writing" / "dataset" / "results",
        baseline_path=PROJECT_ROOT / "creative_writing" / "dataset" / "results"
                      / "claude_sonnet_4.6_result.csv",
        categories=CW_CATEGORIES,
        id_keys=["index", "model"],
        pretty_name=_cw_pretty,
        glob_pattern="*_result.csv",
        score_range=20.0,
    ),
]


# =============================================================================
# Shared computation
# =============================================================================

def load_dataset(cfg: DatasetConfig):
    """Returns (pairs, errors, evaluator_names)."""
    raw = {p.stem: pd.read_csv(p) for p in sorted(cfg.results_dir.glob(cfg.glob_pattern))}

    baseline_df = None
    if cfg.baseline_path is not None:
        baseline_df = pd.read_csv(cfg.baseline_path)
        raw.pop(cfg.baseline_path.stem, None)  # don't double-count the baseline

    evaluators = {cfg.pretty_name(k): v for k, v in raw.items()}
    evaluator_names = sorted(evaluators.keys())

    pair_frames, err_frames = [], []
    for name, df in evaluators.items():
        if baseline_df is not None:
            m = df.merge(baseline_df, on=cfg.id_keys, suffixes=("_ev", "_ref"),
                         validate="one_to_one")
            for cat in cfg.categories:
                col = f"{cat}_score"
                ref = m[f"{col}_ref"].to_numpy()
                sc = m[f"{col}_ev"].to_numpy()
                pair_frames.append(pd.DataFrame({"evaluator": name, "category": cat,
                                                 "baseline": ref, "score": sc}))
                block = m[cfg.id_keys].copy()
                block["category"] = cat
                block["evaluator"] = name
                block["abs_err"] = np.abs(ref - sc)
                block["sq_err"] = (ref - sc) ** 2
                err_frames.append(block)
        else:
            for cat in cfg.categories:
                ref = df[cfg.ref_col_for(cat)].to_numpy()
                sc = df[f"{cat}_score"].to_numpy()
                pair_frames.append(pd.DataFrame({"evaluator": name, "category": cat,
                                                 "baseline": ref, "score": sc}))
                block = df[cfg.id_keys].copy()
                block["category"] = cat
                block["evaluator"] = name
                block["abs_err"] = np.abs(ref - sc)
                block["sq_err"] = (ref - sc) ** 2
                err_frames.append(block)

    return (pd.concat(pair_frames, ignore_index=True),
            pd.concat(err_frames, ignore_index=True),
            evaluator_names)


def bootstrap_calibration(pairs: pd.DataFrame, k: int, alpha: float,
                          seed: int) -> pd.DataFrame:
    """For each (evaluator, category) group draw k subsamples of size m=floor(sqrt(n))
    with replacement and compute MAE / RMSE / Spearman ρ / Kendall τ summaries."""
    rng = np.random.default_rng(seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="An input array is constant")
        for (ev, cat), g in pairs.groupby(["evaluator", "category"], sort=False):
            x = g["baseline"].to_numpy()
            y = g["score"].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            n = len(x)
            if n < 2:
                continue
            m = max(2, int(np.floor(np.sqrt(n))))
            mae = np.empty(k)
            rmse = np.empty(k)
            rho = np.full(k, np.nan)
            kendall = np.full(k, np.nan)
            for j in range(k):
                idx = rng.integers(0, n, size=m)
                xs, ys = x[idx], y[idx]
                diff = xs - ys
                mae[j] = np.mean(np.abs(diff))
                rmse[j] = float(np.sqrt(np.mean(diff ** 2)))
                if xs.min() != xs.max() and ys.min() != ys.max():
                    r = stats.spearmanr(xs, ys).correlation
                    if np.isfinite(r):
                        rho[j] = r
                    t = stats.kendalltau(xs, ys).correlation
                    if np.isfinite(t):
                        kendall[j] = t
            has_rho = np.any(~np.isnan(rho))
            has_tau = np.any(~np.isnan(kendall))
            rows.append({
                "evaluator": ev, "category": cat, "n": n, "m": m,
                "mean_mae": float(np.mean(mae)),
                "ucb_mae": float(np.quantile(mae, 1 - alpha)),
                "mean_rmse": float(np.mean(rmse)),
                "ucb_rmse": float(np.quantile(rmse, 1 - alpha)),
                "mean_rho": float(np.nanmean(rho)) if has_rho else np.nan,
                "lcb_rho": float(np.nanquantile(rho, alpha)) if has_rho else np.nan,
                "mean_kendall": float(np.nanmean(kendall)) if has_tau else np.nan,
                "lcb_kendall": float(np.nanquantile(kendall, alpha)) if has_tau else np.nan,
            })
    return pd.DataFrame(rows)


def compute_cat_corr(pairs, categories, evaluator_names,
                     method: str = "spearman") -> pd.DataFrame:
    """Per (evaluator, category) rank correlation on the full validation set.

    method: 'spearman' (Spearman ρ) or 'kendall' (Kendall τ).
    """
    fn = stats.spearmanr if method == "spearman" else stats.kendalltau
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="An input array is constant")
        rows = []
        for (ev, cat), g in pairs.groupby(["evaluator", "category"]):
            x = g["baseline"].to_numpy(); y = g["score"].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 2 or x.min() == x.max() or y.min() == y.max():
                r = np.nan
            else:
                r = float(fn(x, y).correlation)
            rows.append({"evaluator": ev, "category": cat, "corr": r})
    return (pd.DataFrame(rows)
            .pivot_table(index="category", columns="evaluator", values="corr")
            .reindex(index=categories, columns=evaluator_names))


def compute_cat_rho(pairs, categories, evaluator_names) -> pd.DataFrame:
    """Per (evaluator, category) Spearman ρ on the full validation set."""
    return compute_cat_corr(pairs, categories, evaluator_names, method="spearman")


def metric_spec(metric: str):
    """Routing-config bundle per metric: score_col, frontier_value, higher_is_better."""
    if metric == "mae":
        return {"score_col": "ucb_mae", "frontier_value": 0.0, "higher_is_better": False}
    if metric == "rmse":
        return {"score_col": "ucb_rmse", "frontier_value": 0.0, "higher_is_better": False}
    if metric == "rho":
        return {"score_col": "lcb_rho", "frontier_value": 1.0, "higher_is_better": True}
    if metric == "kendall":
        return {"score_col": "lcb_kendall", "frontier_value": 1.0, "higher_is_better": True}
    raise ValueError(f"Unknown metric: {metric}")


def compute_cat_value(pairs, errors, metric, categories, evaluator_names):
    """Full-data per (rubric × evaluator) value table, used as the realized-quality
    lookup once routing decisions are made."""
    if metric == "mae":
        return (errors.groupby(["evaluator", "category"])["abs_err"].mean()
                .unstack("evaluator")
                .reindex(index=categories, columns=evaluator_names))
    if metric == "rmse":
        return (errors.groupby(["evaluator", "category"])["sq_err"].mean()
                .pow(0.5)
                .unstack("evaluator")
                .reindex(index=categories, columns=evaluator_names))
    if metric == "rho":
        return compute_cat_corr(pairs, categories, evaluator_names, method="spearman")
    if metric == "kendall":
        return compute_cat_corr(pairs, categories, evaluator_names, method="kendall")
    raise ValueError(f"Unknown metric: {metric}")


def operating_point_at_threshold(score_table, value_lookup, categories,
                                 calib_calls, n_per_cat, higher_is_better, threshold,
                                 debug=False, debug_tag=""):
    """Single operating point for a user-supplied fallback threshold T.

    Policy: for each category, route via the best surrogate iff its calibrated
    score clears T; otherwise fall back / escalate that whole category to the
    frontier model. For higher-is-better metrics (rho) "clears" means score >= T;
    for lower-is-better (mae) it means score <= T. Pass +/-inf to disable
    fallback entirely (surrogate-only operating point).
    """
    if higher_is_better:
        chosen = score_table.idxmax(axis=1)
        chosen_score = score_table.max(axis=1)
        frontier_value = 1.0
        fails = lambda s: s < threshold
        cmp = "<"
    else:
        chosen = score_table.idxmin(axis=1)
        chosen_score = score_table.min(axis=1)
        frontier_value = 0.0
        fails = lambda s: s > threshold
        cmp = ">"

    valid = chosen.dropna()
    chosen_value = pd.Series({c: value_lookup.loc[c, valid[c]] for c in valid.index})
    realized = chosen_value.copy()
    n_escalated = 0
    escalated_cats = []
    for cat in valid.index:
        if fails(chosen_score[cat]):
            realized.loc[cat] = frontier_value
            n_escalated += 1
            escalated_cats.append(cat)

    if debug:
        header = f"[debug{(' ' + debug_tag) if debug_tag else ''}] threshold={threshold!r}"
        print(header)
        print(f"  policy: escalate rubric if calibrated_score {cmp} {threshold!r}")
        rows = []
        for cat in valid.index:
            cs = float(chosen_score[cat])
            cv = float(chosen_value[cat])
            rows.append({
                "rubric": cat,
                "best_surrogate": valid[cat],
                "calibrated_score": cs,
                "fails": fails(cs),
                "full_data_score": cv,
                "realized": frontier_value if fails(cs) else cv,
            })
        df = pd.DataFrame(rows)
        with pd.option_context("display.max_rows", None,
                               "display.max_colwidth", 60,
                               "display.width", 140,
                               "display.float_format", lambda v: f"{v: .4f}"):
            print(df.to_string(index=False))
        print(f"  -> n_escalated = {n_escalated}/{len(valid)}  "
              f"escalated = {escalated_cats}")
        print(f"  -> calls = {calib_calls + n_escalated * n_per_cat}  "
              f"metric = {float(realized.mean()):.4f}")
        print()

    return {
        "calls": calib_calls + n_escalated * n_per_cat,
        "metric": float(realized.mean()),
        "threshold": float(threshold),
        "n_escalated": int(n_escalated),
    }


def per_instance_oracle_error_curve(per_q_min, categories, error_kind="mae",
                                    calib_calls=0, downsample_to=400):
    """Greedily escalate highest-|error| (story, cat) pairs to frontier.

    error_kind:
        'mae'  → macro-avg of per-rubric mean absolute error  (uses abs_err)
        'rmse' → macro-avg of per-rubric sqrt-mean-squared err (uses sq_err)

    Both versions sort by abs_err (since |err| ≥ 0, ranking by abs_err is the
    same as ranking by sq_err — the squared-error oracle is identical in which
    pairs to escalate, only the y-axis aggregation differs).
    """
    err_col = "abs_err" if error_kind == "mae" else "sq_err"
    df = per_q_min.sort_values("abs_err", ascending=False).reset_index(drop=True)
    cat_total_count = (per_q_min.groupby("category").size()
                       .reindex(categories).to_numpy())
    cat_total_sum = (per_q_min.groupby("category")[err_col].sum()
                     .reindex(categories).to_numpy())
    cat_dummies = (pd.get_dummies(df["category"])
                   .reindex(columns=categories, fill_value=0).to_numpy())
    cum = np.cumsum(cat_dummies * df[err_col].to_numpy()[:, None], axis=0)
    cum = np.vstack([np.zeros(len(categories)), cum])
    per_cat_mean = (cat_total_sum - cum) / cat_total_count
    if error_kind == "rmse":
        per_cat_mean = np.sqrt(np.maximum(per_cat_mean, 0.0))
    macro = per_cat_mean.mean(axis=1)
    inf = np.arange(len(df) + 1)
    out = pd.DataFrame({"calls": calib_calls + inf, "metric": macro})
    if len(out) > downsample_to:
        idx = np.unique(np.linspace(0, len(out) - 1, downsample_to).astype(int))
        out = out.iloc[idx].reset_index(drop=True)
    return out


def per_instance_oracle_mae_curve(per_q_min, categories, calib_calls=0, downsample_to=400):
    """Backward-compat alias for the MAE per-instance oracle curve."""
    return per_instance_oracle_error_curve(
        per_q_min, categories, error_kind="mae",
        calib_calls=calib_calls, downsample_to=downsample_to,
    )


# =============================================================================
# Plotting
# =============================================================================

STAT_STYLES = {
    5:   {"color": "#2E86AB"},  # deep blue
    10:  {"color": "#A23B72"},  # magenta
    50:  {"color": "#F18F01"},  # warm orange
    500: {"color": "#3B1F2B"},  # near-black plum
}

# Marker shapes cycle per operating point so colliding points stay distinguishable.
POINT_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def _marker_size(i: int) -> int:
    """Shrink markers as we walk along a k's trajectory so layered/colliding
    points (across k's, or with single-evaluator dots) stay readable."""
    # return max(70, 230 - i * 32)
    return 30


def make_figure(cfg, metric, k_values, alpha, seed, tolerances=(), debug=False):
    pairs, errors, evaluator_names = load_dataset(cfg)

    per_q_min = errors.groupby(cfg.id_keys + ["category"])["abs_err"].min().reset_index()
    per_q_min["sq_err"] = per_q_min["abs_err"] ** 2
    n_per_cat = len(per_q_min) // len(cfg.categories)
    n_inference = len(per_q_min)

    n_pairs_per_group = int(pairs.groupby(["evaluator", "category"]).size().iloc[0])
    m_calib = max(2, int(np.floor(np.sqrt(n_pairs_per_group))))
    calib_calls = {k: m_calib * k for k in k_values}

    higher_is_better = metric in ("rho", "kendall")
    is_error_metric = metric in ("mae", "rmse")

    if metric == "mae":
        cat_value = (errors.groupby(["evaluator", "category"])["abs_err"].mean()
                     .unstack("evaluator")
                     .reindex(index=cfg.categories, columns=evaluator_names))
        score_col = "ucb_mae"
        frontier_value = 0.0
        ylabel = "MAE"
        title = f"Cost-quality Pareto — {cfg.label}  (MAE)"
    elif metric == "rmse":
        cat_value = (errors.groupby(["evaluator", "category"])["sq_err"].mean()
                     .pow(0.5)
                     .unstack("evaluator")
                     .reindex(index=cfg.categories, columns=evaluator_names))
        score_col = "ucb_rmse"
        frontier_value = 0.0
        ylabel = "RMSE"
        title = f"Cost-quality Pareto — {cfg.label}  (RMSE)"
    elif metric == "rho":
        cat_value = compute_cat_corr(pairs, cfg.categories, evaluator_names,
                                     method="spearman")
        score_col = "lcb_rho"
        frontier_value = 1.0
        ylabel = "Mean Spearman ρ"
        title = f"Cost-quality Pareto — {cfg.label}  (Spearman ρ)"
    elif metric == "kendall":
        cat_value = compute_cat_corr(pairs, cfg.categories, evaluator_names,
                                     method="kendall")
        score_col = "lcb_kendall"
        frontier_value = 1.0
        ylabel = "Mean Kendall τ"
        title = f"Cost-quality Pareto — {cfg.label}  (Kendall τ)"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    score_tables = {}
    for k in k_values:
        cal = bootstrap_calibration(pairs, k=k, alpha=alpha, seed=seed)
        score_tables[k] = (cal.pivot_table(index="category", columns="evaluator",
                                           values=score_col)
                           .reindex(index=cfg.categories, columns=evaluator_names))

    # MoJ-stat is now plotted only at user-supplied τ (further down). The
    # per-instance oracle is a true per-pair sweep so it stays a line.
    pi_oracle_curve = None
    if is_error_metric:
        pi_oracle_curve = per_instance_oracle_error_curve(
            per_q_min, cfg.categories, error_kind=metric,
        )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Single-evaluator macro baselines at x=0. Soft gray so the colored MoJ-stat
    # markers stay the visual focus; small horizontal jitter on label placement
    # so vertically-close evaluators don't stack their text on top of each other.
    single_macro = cat_value.mean(axis=0).sort_values(ascending=not higher_is_better)
    for j, (ev, y) in enumerate(single_macro.items()):
        y = float(y)
        ax.scatter([0], [y], s=55, color="#9aa3ad", edgecolor="white", linewidth=0.0,
                   zorder=4)
        ax.annotate(f"{ev} ({y:.2f})", (0, y),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8, va="center", color="#4a5560")

    # MoJ-stat operating points: per k, dedupe thresholds that map to the same
    # n_escalated and merge their descriptions. Connect surviving points with a
    # subtle dashed trajectory so the "tighten T → escalate more → climb" story
    # is visible. Marker size shrinks along the curve so overlapping points
    # (across k's, or with the gray evaluator dots at x=0) layer cleanly.
    no_fallback_threshold = float("-inf") if higher_is_better else float("inf")
    for k in k_values:
        color = STAT_STYLES.get(k, {"color": "#555555"})["color"]

        specs = [("No fallback", no_fallback_threshold)]
        for thr in tolerances:
            # Error metrics (MAE/RMSE) take thresholds as a fraction of the
            # rubric's evaluation range (0.05 = 5%). Correlation metrics (ρ/τ)
            # take thresholds as absolute values, since they live on [-1, 1].
            abs_thr = thr * cfg.score_range if is_error_metric else thr
            if metric == "mae":
                desc = f"MAE≤{abs_thr:.2f} ({thr * 100:g}% of range)"
            elif metric == "rmse":
                desc = f"RMSE≤{abs_thr:.2f} ({thr * 100:g}% of range)"
            elif metric == "rho":
                desc = f"ρ≥{thr:g}"
            elif metric == "kendall":
                desc = f"τ≥{thr:g}"
            else:
                desc = f"T={thr:g}"
            specs.append((desc, abs_thr))

        if debug:
            print(f"\n=== {cfg.name} / metric={metric} / k={k} "
                  f"(score_col={score_col}, alpha={alpha}, seed={seed}) ===")
            print(f"calibrated score_table (per rubric × evaluator):")
            with pd.option_context("display.max_rows", None,
                                   "display.max_colwidth", 30,
                                   "display.width", 160,
                                   "display.float_format", lambda v: f"{v: .4f}"):
                print(score_tables[k].to_string())
            print()

        by_n = {}  # n_escalated → (op, [descs])
        for desc, abs_thr in specs:
            op = operating_point_at_threshold(
                score_tables[k], cat_value, cfg.categories,
                0, n_per_cat, higher_is_better, abs_thr,
                debug=debug,
                debug_tag=f"{cfg.name}/{metric}/k={k}/{desc}",
            )
            n = op["n_escalated"]
            if n in by_n:
                by_n[n][1].append(desc)
            else:
                by_n[n] = (op, [desc])

        sorted_pts = sorted(by_n.items(), key=lambda kv: kv[0])

        # Plot largest markers first so smaller ones land on top and stay visible.
        for i, (n, (op, descs)) in enumerate(sorted_pts):
            marker = POINT_MARKERS[i % len(POINT_MARKERS)]
            merged = " = ".join(descs)
            # label = (f"k={k} (+{calib_calls[k]:,} calib)  {merged}  "
            #          f"[{n}/{len(cfg.categories)} esc.]")
            label = (f"{merged}  ")
                    #  f"[{n}/{len(cfg.categories)} esc.]")
            ax.scatter([op["calls"]], [op["metric"]], marker=marker,
                       color=color, edgecolor="white", linewidth=1,
                       s=_marker_size(i), alpha=0.92, zorder=7 + i, label=label)

    if pi_oracle_curve is not None:
        ax.plot(pi_oracle_curve["calls"], pi_oracle_curve["metric"],
                "-", color="#444444", lw=1.8, alpha=0.85,
                label="Optimal routing",
                zorder=3)

    ax.scatter([n_inference], [frontier_value], s=260, color="#1a1a1a", marker="*",
               edgecolor="white", linewidth=0, zorder=8,
               label=f"Pure frontier")

    ax.set_xlabel("Cost - # of (ans, rubric) pairs\ngraded by frontier model")
    ax.set_ylabel("Quality - " + ylabel)

    # Bold main title above a smaller-grey subtitle that explains the operating points.
    # fig.suptitle(title, fontweight="bold", fontsize=12, y=0.985)
    # ax.set_title(
    #     "Each MoJ-stat marker is one operating point: categories whose calibrated\n"
    #     "quality fails the fallback threshold T are routed to the frontier model.",
    #     fontsize=9, color="dimgray"
    # )
    ax.set_title(cfg.label)

    if higher_is_better:
        ax.set_ylim(top=1.05)
    else:
        ax.set_ylim(bottom=-0.05)
    ax.legend(fontsize=8)
    ax.set_xlim(left=-max(8, n_inference * 0.01))
    ax.grid(True, alpha=0.25, linewidth=0.6)
            #   bbox_to_anchor=(1.02, 0.5),

    # Footnote: clarifies that calibration cost is real but not on the x-axis.
    # fig.text(
    #     0.99, 0.005,
    #     "MoJ-stat also incurs an upfront calibration cost (per k, in legend); "
    #     "not shown on the x-axis.",
    #     ha="right", va="bottom", fontsize=8, color="gray", style="italic",
    # )

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
    p.add_argument("--k-values", nargs="+", type=int, default=[5],
                   help="Bootstrap k values to overlay (default: 5 10 50)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Confidence level for UCB/LCB (default: 0.05)")
    p.add_argument("--seed", type=int, default=0,
                   help="Bootstrap seed (default: 0)")
    p.add_argument("--tolerance-mae", nargs="*", type=float, default=[],
                   help="MAE fallback thresholds, as a fraction of the rubric's evaluation "
                        "range (e.g. 0.05 = 5%%). Rubrics with calibrated MAE > T*range "
                        "fall back to the frontier model.")
    p.add_argument("--tolerance-rmse", nargs="*", type=float, default=[],
                   help="RMSE fallback thresholds, as a fraction of the rubric's evaluation "
                        "range. Rubrics with calibrated RMSE > T*range fall back.")
    p.add_argument("--tolerance-rho", nargs="*", type=float, default=[],
                   help="Spearman ρ fallback thresholds T (absolute, since ρ ∈ [-1, 1]); "
                        "rubrics with calibrated ρ < T fall back to the frontier model.")
    p.add_argument("--tolerance-kendall", nargs="*", type=float, default=[],
                   help="Kendall τ fallback thresholds T (absolute, since τ ∈ [-1, 1]); "
                        "rubrics with calibrated τ < T fall back to the frontier model.")
    p.add_argument("--debug", action="store_true",
                   help="Print the calibrated per-rubric × evaluator score table and the "
                        "per-threshold routing decisions for each (dataset, metric, k).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        for metric in args.metrics:
            print(f"[{cfg.name} / {metric}] generating ...")
            tolerances = {
                "mae": args.tolerance_mae,
                "rmse": args.tolerance_rmse,
                "rho": args.tolerance_rho,
                "kendall": args.tolerance_kendall,
            }[metric]
            fig = make_figure(cfg, metric=metric, k_values=args.k_values,
                              alpha=args.alpha, seed=args.seed,
                              tolerances=tolerances, debug=args.debug)
            base = args.output_dir / f"pareto_{cfg.name}_{metric}"
            fig.savefig(f"{base}.pdf", bbox_inches="tight")
            fig.savefig(f"{base}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  saved {base.name}.pdf / .png")


if __name__ == "__main__":
    main()
