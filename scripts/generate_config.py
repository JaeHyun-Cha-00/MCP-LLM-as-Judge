"""
Generate routing_config.json for routed_mcp_server.py.

For each rubric in a dataset, determines the best open-weight judge model
using bootstrap-calibrated UCB-MAE (or mean Spearman ρ) computed from the
MOJ benchmark result CSVs.

Usage:
    # Derive assignments from result CSVs (default: UCB-MAE):
    python generate_config.py \\
        --registry ../configs/model_registry.json \\
        --output   ../configs/routing_config.json

    # Use Spearman ρ instead of MAE:
    python generate_config.py --metric rho --output ../configs/routing_config.json

    # Apply a quality threshold τ — rubrics where no model beats τ fall back
    # to the baseline (claude-sonnet-4-6):
    python generate_config.py --metric mae --tau 1.5 --output ../configs/routing_config.json

    # Print summary table without writing a file:
    python generate_config.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved at runtime)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "creative_writing": {
        "description": "Evaluate creative writing stories using EQ-Bench v3 rubrics (0-20 scale).",
        "results_dir": REPO_ROOT / "creative_writing" / "dataset" / "results",
        "baseline_stem": "claude_sonnet_4.6",
        "data_path": "creative_writing/dataset/data.csv",
        "text_column": "story",
        "hf_id": None,
        "size": 767,
        "score_type": "float",
        "clamp_min": 0.0,
        "clamp_max": 20.0,
        # rubric_id → (type, ref_source)
        # ref_source="baseline_file" means scores come from baseline CSV on same index
        "rubrics": [
            ("Adherence to Instructions",          "positive", "baseline_file"),
            ("Believable Character Actions",        "positive", "baseline_file"),
            ("Nuanced Characters",                  "positive", "baseline_file"),
            ("Consistent Voice / Tone of Writing",  "positive", "baseline_file"),
            ("Imagery and Descriptive Quality",     "positive", "baseline_file"),
            ("Elegant Prose",                       "positive", "baseline_file"),
            ("Emotionally Engaging",                "positive", "baseline_file"),
            ("Emotionally Complex",                 "positive", "baseline_file"),
            ("Coherent",                            "positive", "baseline_file"),
            ("Well-earned Lightness or Darkness",   "positive", "baseline_file"),
            ("Sentences Flow Naturally",            "positive", "baseline_file"),
            ("Overall Reader Engagement",           "positive", "baseline_file"),
            ("Overall Impression",                  "positive", "baseline_file"),
            ("Meandering",                          "negative", "baseline_file"),
            ("Weak Dialogue",                       "negative", "baseline_file"),
            ("Tell-Don't-Show",                     "negative", "baseline_file"),
            ("Unsurprising or Uncreative",          "negative", "baseline_file"),
            ("Amateurish",                          "negative", "baseline_file"),
            ("Purple Prose",                        "negative", "baseline_file"),
            ("Overwrought",                         "negative", "baseline_file"),
            ("Incongruent Ending Positivity",       "negative", "baseline_file"),
            ("Unearned Transformations",            "negative", "baseline_file"),
        ],
    },
    "story_writing": {
        "description": "Evaluate story quality using lars1234/story_writing_benchmark rubrics (0-5 scale).",
        "results_dir": REPO_ROOT / "story_writing_benchmark" / "dataset" / "results",
        "baseline_stem": "claude_sonnet_4.6",
        "data_path": "story_writing_benchmark/dataset/data.csv",
        "text_column": "story",
        "hf_id": "lars1234/story_writing_benchmark",
        "size": 3480,
        "score_type": "int",
        "clamp_min": 0,
        "clamp_max": 5,
        # ref_source="ref_col" means the column is already in the evaluator CSV
        "rubrics": [
            ("Grammar, Spelling, and Punctuation Quality",  "positive", "ref_q1"),
            ("Clarity and Understandability",               "positive", "ref_q2"),
            ("Logical Connection Between Events and Ideas", "positive", "ref_q3"),
            ("Scene Construction and Purpose",              "positive", "ref_q4"),
            ("Internal Consistency",                        "positive", "ref_q5"),
            ("Character Consistency",                       "positive", "ref_q6"),
            ("Character Motivation and Actions",            "positive", "ref_q7"),
            ("Sentence Pattern Variety",                    "positive", "ref_q8"),
            ("Avoidance of Clichés and Overused Phrases",   "positive", "ref_q9"),
            ("Natural Dialogue",                            "positive", "ref_q10"),
            ("Avoidance of Predictable Narrative Tropes",   "positive", "ref_q11"),
            ("Character Depth and Dimensionality",          "positive", "ref_q12"),
            ("Realistic Character Interactions",            "positive", "ref_q13"),
            ("Ability to Hold Reader Interest",             "positive", "ref_q14"),
            ("Satisfying Plot Resolution",                  "positive", "ref_q15"),
        ],
    },
}


# ---------------------------------------------------------------------------
# MOJ bootstrap calibration
# ---------------------------------------------------------------------------

def _bootstrap_ucb_mae(
    errors: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> float:
    """Upper confidence bound of MAE via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(errors)
    boot_means = np.array([
        rng.choice(errors, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.percentile(boot_means, ci * 100))


def _bootstrap_lcb_rho(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> float:
    """Lower confidence bound of Spearman ρ via percentile bootstrap."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(rho)
    return float(np.percentile(boot_rhos, (1 - ci) * 100))


def compute_calibration(
    results_dir: Path,
    rubric_ids: list[str],
    baseline_stem: str,
    candidate_stems: list[str],           # non-baseline model stems
    ref_source_map: dict[str, str],       # rubric_id → ref_source string
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        model_stem | rubric | ucb_mae | lcb_rho | mean_mae | mean_rho | n

    For creative_writing: baseline scores come from the baseline CSV (matched on index).
    For story_writing:    baseline scores are the ref_q* columns already in each CSV.
    """
    score_cols = [f"{r}_score" for r in rubric_ids]
    use_baseline_file = any(v == "baseline_file" for v in ref_source_map.values())

    # Load baseline scores (creative_writing only)
    baseline_scores: pd.DataFrame | None = None
    if use_baseline_file:
        baseline_path = results_dir / f"{baseline_stem}_result.csv"
        baseline_df = pd.read_csv(baseline_path, index_col="index")
        baseline_scores = baseline_df[score_cols].copy()

    rows = []
    for stem in candidate_stems:
        path = results_dir / f"{stem}_result.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="index")

        for rubric in rubric_ids:
            score_col = f"{rubric}_score"
            ref_source = ref_source_map[rubric]

            if ref_source == "baseline_file":
                common_idx = df.index.intersection(baseline_scores.index)
                eval_vals = df.loc[common_idx, score_col].values.astype(float)
                ref_vals  = baseline_scores.loc[common_idx, score_col].values.astype(float)
            else:
                # ref_source is a column name in the evaluator CSV (e.g. "ref_q1")
                valid = df[[score_col, ref_source]].dropna()
                eval_vals = valid[score_col].values.astype(float)
                ref_vals  = valid[ref_source].values.astype(float)

            if len(eval_vals) < 10:
                continue

            abs_err = np.abs(eval_vals - ref_vals)
            ucb_mae = _bootstrap_ucb_mae(abs_err, n_bootstrap=n_bootstrap, seed=seed)
            lcb_rho = _bootstrap_lcb_rho(eval_vals, ref_vals, n_bootstrap=n_bootstrap, seed=seed)

            rows.append({
                "model_stem": stem,
                "rubric":     rubric,
                "ucb_mae":    ucb_mae,
                "lcb_rho":    lcb_rho,
                "mean_mae":   float(abs_err.mean()),
                "mean_rho":   float(np.corrcoef(eval_vals, ref_vals)[0, 1]),
                "n":          len(eval_vals),
            })

    return pd.DataFrame(rows)


def assign_models(
    cal: pd.DataFrame,
    metric: str,                          # "mae" or "rho"
    rubric_ids: list[str],
    stem_to_alias: dict[str, str],        # result_stem → model alias
    tau: float | None = None,
    fallback_alias: str = "claude-sonnet-4-6",
) -> dict[str, dict[str, Any]]:
    """
    For each rubric, pick the model with the best calibrated metric.
    Returns {rubric_id: {"judge_model": alias, "ucb_mae": ..., "lcb_rho": ..., "mean_mae": ..., "mean_rho": ...}}
    """
    assignments: dict[str, dict[str, Any]] = {}

    for rubric in rubric_ids:
        sub = cal[cal["rubric"] == rubric]
        if sub.empty:
            assignments[rubric] = {
                "judge_model": fallback_alias,
                "ucb_mae": None, "lcb_rho": None,
                "mean_mae": None, "mean_rho": None,
            }
            continue

        if metric == "mae":
            best_row = sub.loc[sub["ucb_mae"].idxmin()]
            if tau is not None and best_row["ucb_mae"] > tau:
                alias = fallback_alias
            else:
                alias = stem_to_alias.get(best_row["model_stem"], fallback_alias)
        else:  # rho
            best_row = sub.loc[sub["lcb_rho"].idxmax()]
            if tau is not None and best_row["lcb_rho"] < tau:
                alias = fallback_alias
            else:
                alias = stem_to_alias.get(best_row["model_stem"], fallback_alias)

        assignments[rubric] = {
            "judge_model": alias,
            "ucb_mae":  round(float(best_row["ucb_mae"]), 4),
            "lcb_rho":  round(float(best_row["lcb_rho"]), 4),
            "mean_mae": round(float(best_row["mean_mae"]), 4),
            "mean_rho": round(float(best_row["mean_rho"]), 4),
        }

    return assignments


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_routing_config(
    registry: dict[str, Any],
    metric: str = "mae",
    tau: float | None = None,
    fallback_alias: str = "claude-sonnet-4-6",
    n_bootstrap: int = 500,
    seed: int = 42,
    server_name: str = "llm-judge-router",
    version: str = "2.0",
    dry_run: bool = False,
) -> dict[str, Any]:

    stem_to_alias = {v["result_stem"]: alias for alias, v in registry.items()}

    # Strip internal-only fields from the registry section in the output config
    clean_registry = {
        alias: {k: v for k, v in cfg.items() if k != "result_stem"}
        for alias, cfg in registry.items()
    }

    categories: dict[str, Any] = {}

    for dataset_key, ds in DATASET_CONFIGS.items():
        results_dir: Path = ds["results_dir"]
        baseline_stem: str = ds["baseline_stem"]
        rubric_defs: list[tuple] = ds["rubrics"]  # (id, type, ref_source)

        rubric_ids    = [r[0] for r in rubric_defs]
        rubric_types  = {r[0]: r[1] for r in rubric_defs}
        ref_source_map = {r[0]: r[2] for r in rubric_defs}

        # Candidate stems: registry models that have a result file for this dataset
        candidate_stems = [
            cfg["result_stem"]
            for alias, cfg in registry.items()
            if alias != fallback_alias and (results_dir / f"{cfg['result_stem']}_result.csv").exists()
        ]

        print(
            f"[{dataset_key}] computing calibration for {len(candidate_stems)} models "
            f"× {len(rubric_ids)} rubrics (n_bootstrap={n_bootstrap}) ...",
            file=sys.stderr,
        )

        cal = compute_calibration(
            results_dir=results_dir,
            rubric_ids=rubric_ids,
            baseline_stem=baseline_stem,
            candidate_stems=candidate_stems,
            ref_source_map=ref_source_map,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        assignments = assign_models(
            cal=cal,
            metric=metric,
            rubric_ids=rubric_ids,
            stem_to_alias=stem_to_alias,
            tau=tau,
            fallback_alias=fallback_alias,
        )

        if dry_run:
            _print_assignment_table(dataset_key, assignments, cal, stem_to_alias)

        rubric_entries = []
        for rubric_id, rtype, _ in rubric_defs:
            a = assignments[rubric_id]
            entry: dict[str, Any] = {
                "id":          rubric_id,
                "type":        rtype,
                "scale_min":   ds["clamp_min"],
                "scale_max":   ds["clamp_max"],
                "scale_type":  ds["score_type"],
                "judge_model": a["judge_model"],
                "moj_metrics": {
                    "ucb_mae":  a["ucb_mae"],
                    "lcb_rho":  a["lcb_rho"],
                    "mean_mae": a["mean_mae"],
                    "mean_rho": a["mean_rho"],
                },
            }
            rubric_entries.append(entry)

        categories[dataset_key] = {
            "description": ds["description"],
            "dataset": {
                "name":        ds.get("dataset_name", dataset_key),
                "path":        ds["data_path"],
                "hf_id":       ds.get("hf_id"),
                "size":        ds.get("size"),
                "text_column": ds["text_column"],
            },
            "evaluation": {
                "score_type": ds["score_type"],
                "clamp_min":  ds["clamp_min"],
                "clamp_max":  ds["clamp_max"],
                "response_format": "json",
                "response_key":    "scores",
                "routing_metric":  metric,
                "routing_tau":     tau,
            },
            "rubrics": rubric_entries,
        }

    return {
        "server_name":    server_name,
        "version":        version,
        "model_registry": clean_registry,
        "categories":     categories,
    }


def _print_assignment_table(
    dataset: str,
    assignments: dict[str, dict],
    cal: pd.DataFrame,
    stem_to_alias: dict[str, str],
) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset}")
    print(f"{'='*60}")
    print(f"  {'Rubric':<47} {'Judge Model':<20} {'UCB-MAE':>8} {'LCB-ρ':>8}")
    print(f"  {'-'*47} {'-'*20} {'-'*8} {'-'*8}")
    for rubric, info in assignments.items():
        mae = f"{info['ucb_mae']:.3f}" if info["ucb_mae"] is not None else "  n/a"
        rho = f"{info['lcb_rho']:.3f}" if info["lcb_rho"] is not None else "  n/a"
        print(f"  {rubric:<47} {info['judge_model']:<20} {mae:>8} {rho:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-rubric routing_config.json from MOJ results")
    parser.add_argument(
        "--registry",
        default=str(REPO_ROOT / "configs" / "model_registry.json"),
        help="Path to model_registry.json",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rho"],
        default="mae",
        help="Routing metric: 'mae' (UCB-MAE, default) or 'rho' (LCB Spearman ρ)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Quality threshold. For MAE: max tolerable UCB-MAE. For rho: min acceptable LCB-ρ. "
             "Rubrics where no candidate beats the threshold fall back to --fallback.",
    )
    parser.add_argument(
        "--fallback",
        default="claude-sonnet-4-6",
        help="Model alias used when no candidate beats the threshold (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap iterations for calibration (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "configs" / "routing_config.json"),
        help="Output path for routing_config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-rubric assignment table without writing output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.registry) as f:
        raw_registry = json.load(f)
    # Drop comment keys
    registry = {k: v for k, v in raw_registry.items() if not k.startswith("_")}

    config = build_routing_config(
        registry=registry,
        metric=args.metric,
        tau=args.tau,
        fallback_alias=args.fallback,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
