from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .metrics import ALL_METRICS, best_model, compute, within_tolerance


def run(
    results_dir: str,
    baseline_stem: str,
    model_registry: dict,
    eval_prompt: str,
    rubrics: list[str],
    metric: str,
    tolerance: float | None,
    output_path: str,
) -> None:
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose: {sorted(ALL_METRICS)}")

    rdir = Path(results_dir)

    # Build stem → (alias, endpoint_cfg) from the registry
    stem_to_alias: dict[str, str] = {}
    stem_to_cfg: dict[str, dict] = {}
    baseline_cfg: dict | None = None

    for alias, cfg in model_registry.items():
        stem = cfg["result_stem"]
        endpoint = {k: v for k, v in cfg.items() if k != "result_stem"}
        stem_to_alias[stem] = alias
        stem_to_cfg[stem] = endpoint
        if stem == baseline_stem:
            baseline_cfg = endpoint

    if baseline_cfg is None:
        raise ValueError(
            f"baseline_stem {baseline_stem!r} not found in registry. "
            f"Available stems: {list(stem_to_alias)}"
        )

    baseline_path = rdir / f"{baseline_stem}_result.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline result file not found: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path, index_col="index")

    # Load every open-weight result file that exists in results_dir
    open_stems = [s for s in stem_to_alias if s != baseline_stem]
    model_dfs: dict[str, pd.DataFrame] = {}
    for stem in open_stems:
        path = rdir / f"{stem}_result.csv"
        if path.exists():
            model_dfs[stem] = pd.read_csv(path, index_col="index")

    print(
        f"Baseline: {baseline_stem} ({len(baseline_df)} rows)\n"
        f"Open-weight models found: {list(model_dfs)}\n"
        f"metric={metric}, tolerance={tolerance}",
        file=sys.stderr,
    )

    rubric_routing: list[dict] = []

    for rubric in rubrics:
        col = f"{rubric}_score"

        if col not in baseline_df.columns:
            print(
                f"  [warn] '{rubric}': column '{col}' missing in baseline CSV → skip",
                file=sys.stderr,
            )
            continue

        b_series = baseline_df[col].dropna()

        model_metrics: dict[str, float] = {}
        for stem, df in model_dfs.items():
            if col not in df.columns:
                continue
            common = b_series.index.intersection(df[col].dropna().index)
            if len(common) < 2:
                continue
            ys_true = b_series.loc[common].tolist()
            ys_pred = df[col].loc[common].tolist()
            try:
                model_metrics[stem] = compute(metric, ys_true, ys_pred)
            except Exception as exc:
                print(f"  [warn] {stem}/{rubric}: {exc}", file=sys.stderr)

        if not model_metrics:
            rubric_routing.append(_fallback_entry(rubric, metric, baseline_cfg))
            continue

        best_stem, score = best_model(model_metrics, metric)
        best_alias = stem_to_alias[best_stem]
        use_baseline = tolerance is not None and not within_tolerance(metric, score, tolerance)
        endpoint = baseline_cfg if use_baseline else stem_to_cfg[best_stem]

        rubric_routing.append({
            "rubric":       rubric,
            "best_model":   best_alias,
            "metric":       metric,
            "metric_value": round(score, 6),
            "use_baseline": use_baseline,
            "endpoint":     endpoint,
        })

        tag = "baseline (outside Tol)" if use_baseline else best_alias
        print(f"  {rubric}: best={best_alias} {metric}={score:.4f} → {tag}", file=sys.stderr)

    open_weight_models = {stem_to_alias[s]: stem_to_cfg[s] for s in model_dfs}

    config = {
        "server_name":        "mojo-judge",
        "eval_prompt":        eval_prompt,
        "metric":             metric,
        "tolerance":          tolerance,
        "baseline_endpoint":  baseline_cfg,
        "open_weight_models": open_weight_models,
        "rubric_routing":     rubric_routing,
    }
    Path(output_path).write_text(json.dumps(config, indent=2))
    print(f"\nConfig → {output_path}", file=sys.stderr)


def _fallback_entry(rubric: str, metric: str, baseline_cfg: dict) -> dict:
    return {
        "rubric":       rubric,
        "best_model":   "__baseline__",
        "metric":       metric,
        "metric_value": None,
        "use_baseline": True,
        "endpoint":     baseline_cfg,
    }
