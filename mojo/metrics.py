from __future__ import annotations

import math

from scipy import stats

LOWER_IS_BETTER = frozenset({"mae", "rmse"})
HIGHER_IS_BETTER = frozenset({"spearman", "kendall"})
ALL_METRICS = LOWER_IS_BETTER | HIGHER_IS_BETTER


def compute(metric: str, y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    if n < 2:
        raise ValueError(f"Need ≥2 paired samples, got {n}.")
    if metric == "mae":
        return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    if metric == "rmse":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n)
    if metric == "spearman":
        return float(stats.spearmanr(y_true, y_pred).statistic)
    if metric == "kendall":
        return float(stats.kendalltau(y_true, y_pred).statistic)
    raise ValueError(f"Unknown metric {metric!r}. Choose: {sorted(ALL_METRICS)}")


def best_model(scores: dict[str, float], metric: str) -> tuple[str, float]:
    """Return (alias, score) of the best-aligned open-weight model."""
    if metric in LOWER_IS_BETTER:
        return min(scores.items(), key=lambda kv: kv[1])
    return max(scores.items(), key=lambda kv: kv[1])


def within_tolerance(metric: str, score: float, tol: float) -> bool:
    """True when the score is good enough that no baseline fallback is needed."""
    if metric in LOWER_IS_BETTER:
        return score <= tol
    return score >= tol
