from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from .metrics import ALL_METRICS, best_model, compute, within_tolerance


def _parse_scores(raw: str, rubrics: list[str]) -> dict[str, float]:
    def _try(text: str) -> dict:
        try:
            d = json.loads(text)
            return d.get("scores", d) if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return d.get("scores", d) if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                pass
        return {}

    parsed = _try(raw)
    out: dict[str, float] = {}
    for r in rubrics:
        if r in parsed:
            try:
                out[r] = float(parsed[r])
            except (TypeError, ValueError):
                pass
    return out


async def _call(
    endpoint: dict,
    eval_prompt: str,
    rubrics: list[str],
    text: str,
) -> dict[str, float]:
    api_key = os.environ.get(endpoint.get("api_key_env", ""), "EMPTY") or "EMPTY"
    client = AsyncOpenAI(base_url=endpoint["base_url"], api_key=api_key)
    rubric_list = "\n".join(f"- {r}" for r in rubrics)
    user_msg = (
        "Evaluate the following text on these criteria.\n"
        "For each criterion provide a numeric score from 0 to 10 (floats allowed).\n"
        'Respond ONLY with JSON: {"scores": {"Criterion": score, ...}}\n\n'
        f"Criteria:\n{rubric_list}\n\nText:\n{text}"
    )
    try:
        resp = await client.chat.completions.create(
            model=endpoint["model_id"],
            messages=[
                {"role": "system", "content": eval_prompt},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=endpoint.get("max_tokens", 1024),
            temperature=endpoint.get("temperature", 0.0),
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [warn] {endpoint.get('model_id')}: {exc}", file=sys.stderr)
        return {}
    return _parse_scores(raw, rubrics)


async def run(
    dataset_path: str,
    text_column: str,
    eval_prompt: str,
    rubrics: list[str],
    baseline_cfg: dict,
    model_cfgs: dict[str, dict],
    metric: str,
    tolerance: float | None,
    output_path: str,
) -> None:
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose: {sorted(ALL_METRICS)}")

    df = pd.read_csv(dataset_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column {text_column!r} not found. Available: {list(df.columns)}"
        )

    texts = df[text_column].dropna().tolist()
    n = len(texts)
    sample_n = max(2, int(math.isqrt(n)))
    import random
    random.seed(42)
    sample = random.sample(texts, min(sample_n, n))
    print(
        f"Dataset: {n} rows → calibrating on {len(sample)} samples "
        f"(metric={metric}, tolerance={tolerance})",
        file=sys.stderr,
    )

    all_cfgs: dict[str, dict] = {"__baseline__": baseline_cfg, **model_cfgs}

    # all_scores[i] = {alias: {rubric: score}} for sample i
    all_scores: list[dict[str, dict[str, float]]] = []

    for i, text in enumerate(sample, 1):
        print(f"  [{i}/{len(sample)}] calling {len(all_cfgs)} models ...", file=sys.stderr)
        tasks = [_call(cfg, eval_prompt, rubrics, text) for cfg in all_cfgs.values()]
        results = await asyncio.gather(*tasks)
        all_scores.append(dict(zip(all_cfgs.keys(), results)))

    rubric_routing: list[dict] = []

    for rubric in rubrics:
        # Collect per-model paired scores aligned by sample index
        baseline_vals = [
            row["__baseline__"].get(rubric)
            for row in all_scores
        ]

        if sum(v is not None for v in baseline_vals) < 2:
            print(
                f"  [warn] '{rubric}': too few baseline scores → baseline fallback",
                file=sys.stderr,
            )
            rubric_routing.append(_fallback_entry(rubric, metric, baseline_cfg))
            continue

        model_metrics: dict[str, float] = {}
        for alias in model_cfgs:
            pairs = [
                (b, row[alias].get(rubric))
                for b, row in zip(baseline_vals, all_scores)
                if b is not None and row[alias].get(rubric) is not None
            ]
            if len(pairs) < 2:
                continue
            ys_true, ys_pred = zip(*pairs)
            try:
                model_metrics[alias] = compute(metric, list(ys_true), list(ys_pred))
            except Exception as exc:
                print(f"  [warn] {alias}/{rubric}: {exc}", file=sys.stderr)

        if not model_metrics:
            rubric_routing.append(_fallback_entry(rubric, metric, baseline_cfg))
            continue

        alias, score = best_model(model_metrics, metric)
        use_baseline = tolerance is not None and not within_tolerance(metric, score, tolerance)
        endpoint = baseline_cfg if use_baseline else model_cfgs[alias]

        rubric_routing.append({
            "rubric":       rubric,
            "best_model":   alias,
            "metric":       metric,
            "metric_value": round(score, 6),
            "use_baseline": use_baseline,
            "endpoint":     endpoint,
        })

        tag = "baseline (outside Tol)" if use_baseline else alias
        print(
            f"  {rubric}: best={alias} {metric}={score:.4f} → {tag}",
            file=sys.stderr,
        )

    config = {
        "server_name":       "mojo-judge",
        "eval_prompt":       eval_prompt,
        "metric":            metric,
        "tolerance":         tolerance,
        "baseline_endpoint": baseline_cfg,
        "open_weight_models": model_cfgs,
        "rubric_routing":    rubric_routing,
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
