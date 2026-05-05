"""Print the combined LaTeX table: individual models + MOJO rows for both datasets.

CW ρ:  pooled across all scores in each polarity group
SWB ρ: mean of per-category values
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
CW_RESULTS  = Path(__file__).parent.parent / "creative_writing" / "dataset" / "results"
SWB_RESULTS = Path(__file__).parent.parent / "story_writing_benchmark" / "dataset" / "results"

CW_BASELINE  = "claude_sonnet_4.6_result.csv"
SWB_BASELINE = "claude_sonnet_4.6_result.csv"

# ── CW categories ─────────────────────────────────────────────────────────────
POSITIVE_CATEGORIES = [
    "Adherence to Instructions", "Believable Character Actions", "Nuanced Characters",
    "Consistent Voice / Tone of Writing", "Imagery and Descriptive Quality", "Elegant Prose",
    "Emotionally Engaging", "Emotionally Complex", "Coherent",
    "Well-earned Lightness or Darkness", "Sentences Flow Naturally",
    "Overall Reader Engagement", "Overall Impression",
]
NEGATIVE_CATEGORIES = [
    "Meandering", "Weak Dialogue", "Tell-Don't-Show", "Unsurprising or Uncreative",
    "Amateurish", "Purple Prose", "Overwrought",
    "Incongruent Ending Positivity", "Unearned Transformations",
]
ALL_CATEGORIES = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES

# ── SWB categories ────────────────────────────────────────────────────────────
SWB_CATEGORIES = [
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

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _pretty(s):
    return s.replace("_result", "")


def bootstrap_calibration(pairs: pd.DataFrame, k: int, alpha: float, seed: int) -> pd.DataFrame:
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
            for j in range(k):
                idx = rng.integers(0, n, size=m)
                mae[j] = np.mean(np.abs(x[idx] - y[idx]))
            rows.append({"evaluator": ev, "category": cat,
                         "ucb_mae": float(np.quantile(mae, 1 - alpha))})
    return pd.DataFrame(rows)


# ── Load CW data ──────────────────────────────────────────────────────────────
def _load_cw_pairs():
    id_keys  = ["index", "model"]
    raw      = {p.stem: pd.read_csv(p) for p in sorted(CW_RESULTS.glob("*_result.csv"))}
    baseline = pd.read_csv(CW_RESULTS / CW_BASELINE)
    raw.pop(Path(CW_BASELINE).stem, None)

    evaluators      = {_pretty(k): v for k, v in raw.items()}
    evaluator_names = sorted(evaluators.keys())

    pair_frames, err_frames = [], []
    for name, df in evaluators.items():
        merged = df.merge(baseline, on=id_keys, suffixes=("_ev", "_ref"), validate="one_to_one")
        for cat in ALL_CATEGORIES:
            col = f"{cat}_score"
            ref = merged[f"{col}_ref"].to_numpy()
            sc  = merged[f"{col}_ev"].to_numpy()
            pair_frames.append(pd.DataFrame({
                "evaluator": name, "category": cat,
                "baseline": ref, "score": sc,
            }))
            block = merged[id_keys].copy()
            block["category"] = cat
            block["evaluator"] = name
            block["abs_err"]   = np.abs(ref - sc)
            err_frames.append(block)

    return (pd.concat(pair_frames, ignore_index=True),
            pd.concat(err_frames, ignore_index=True),
            evaluator_names)


# ── Load SWB data ─────────────────────────────────────────────────────────────
def _load_swb_pairs():
    swb_keys = ["index", "prompt_id", "model"]
    raw      = {p.stem: pd.read_csv(p) for p in sorted(SWB_RESULTS.glob("*_result.csv"))}
    baseline = pd.read_csv(SWB_RESULTS / SWB_BASELINE)
    raw.pop("claude_sonnet_4.6_result", None)

    evaluators = {_pretty(k): v for k, v in raw.items()}
    ev_names   = sorted(evaluators.keys())

    pair_frames, err_frames = [], []
    for name, df in evaluators.items():
        merged = df.merge(baseline, on=swb_keys, suffixes=("_ev", "_ref"), validate="one_to_one")
        for cat in SWB_CATEGORIES:
            col = f"{cat}_score"
            ref = merged[f"{col}_ref"].to_numpy()
            sc  = merged[f"{col}_ev"].to_numpy()
            pair_frames.append(pd.DataFrame({
                "evaluator": name, "category": cat,
                "baseline": ref, "score": sc,
            }))
            block = merged[swb_keys].copy()
            block["category"]  = cat
            block["evaluator"] = name
            block["abs_err"]   = np.abs(ref - sc)
            err_frames.append(block)

    return (pd.concat(pair_frames, ignore_index=True),
            pd.concat(err_frames, ignore_index=True),
            ev_names)


# ── MOJO helpers ──────────────────────────────────────────────────────────────
def _mojo_mae(rubrics, ucb_tbl, candidates, mae_tbl, thr):
    maes = []
    for rubric in rubrics:
        eligible = sorted([(ucb_tbl.loc[rubric, m], m) for m in candidates
                           if ucb_tbl.loc[rubric, m] <= thr])
        maes.append(0.0 if not eligible else float(mae_tbl.loc[rubric, eligible[0][1]]))
    return float(np.mean(maes))


def _mojo_cw_pooled_corr(rubric_list, ucb_tbl, candidates, pairs_df, thr):
    all_x, all_y = [], []
    for rubric in rubric_list:
        eligible = sorted([(ucb_tbl.loc[rubric, m], m) for m in candidates
                           if ucb_tbl.loc[rubric, m] <= thr])
        if not eligible:
            ref_vals = pairs_df[(pairs_df["category"] == rubric) &
                                (pairs_df["evaluator"] == candidates[0])]["baseline"].dropna().to_numpy()
            all_x.extend(ref_vals)
            all_y.extend(ref_vals)
        else:
            _, m = eligible[0]
            g = pairs_df[(pairs_df["category"] == rubric) & (pairs_df["evaluator"] == m)]
            x = g["baseline"].to_numpy()
            y = g["score"].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            all_x.extend(x[mask])
            all_y.extend(y[mask])
    x = np.array(all_x)
    y = np.array(all_y)
    return float(stats.spearmanr(x, y)[0]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else 1.0


def _mojo_swb_corr(rubrics, ucb_tbl, candidates, mae_tbl, rho_tbl, thr):
    maes, rhos = [], []
    for rubric in rubrics:
        eligible = sorted([(ucb_tbl.loc[rubric, m], m) for m in candidates
                           if ucb_tbl.loc[rubric, m] <= thr])
        if not eligible:
            mae_v, rho_v = 0.0, 1.0
        else:
            _, m = eligible[0]
            mae_v = float(mae_tbl.loc[rubric, m])
            rho_v = float(rho_tbl.loc[rubric, m])
        maes.append(mae_v)
        rhos.append(rho_v)
    return float(np.mean(maes)), float(np.mean(rhos))


def _cw_pooled_corr(pairs, model, rubric_list):
    g = pairs[(pairs["evaluator"] == model) & (pairs["category"].isin(rubric_list))]
    x = g["baseline"].to_numpy()
    y = g["score"].to_numpy()
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    return float(stats.spearmanr(x, y)[0]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


def _cell(val):
    if val is None:
        return "---"
    return f"$-${abs(val):.3f}" if val < 0 else f"{val:.3f}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # CW
    cw_pairs, cw_errors, cw_ev_names = _load_cw_pairs()
    cw_cal = bootstrap_calibration(cw_pairs, k=5, alpha=0.05, seed=0)
    score_table = (
        cw_cal.pivot_table(index="category", columns="evaluator", values="ucb_mae")
              .reindex(index=ALL_CATEGORIES, columns=cw_ev_names)
    )
    cat_mae = (
        cw_errors.groupby(["evaluator", "category"])["abs_err"].mean()
                 .unstack("evaluator")
                 .reindex(index=ALL_CATEGORIES, columns=cw_ev_names)
    )

    # SWB
    swb_pairs, swb_errors, swb_ev_names = _load_swb_pairs()
    swb_cat_mae = (
        swb_errors.groupby(["evaluator", "category"])["abs_err"].mean()
                  .unstack("evaluator")
                  .reindex(index=SWB_CATEGORIES, columns=swb_ev_names)
    )

    swb_corr_rows = []
    for (ev, cat), g in swb_pairs.groupby(["evaluator", "category"]):
        x = g["baseline"].to_numpy()
        y = g["score"].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        rho = float(stats.spearmanr(x, y)[0]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        swb_corr_rows.append({"evaluator": ev, "category": cat, "rho": rho})

    swb_corr_df   = pd.DataFrame(swb_corr_rows)
    swb_rho_table = (
        swb_corr_df.pivot_table(index="category", columns="evaluator", values="rho")
                   .reindex(index=SWB_CATEGORIES, columns=swb_ev_names)
    )

    swb_cal = bootstrap_calibration(swb_pairs, k=5, alpha=0.05, seed=0)
    swb_score_table = (
        swb_cal.pivot_table(index="category", columns="evaluator", values="ucb_mae")
               .reindex(index=SWB_CATEGORIES, columns=swb_ev_names)
    )

    # Build table rows
    MODEL_ORDER = [
        "Qwen3.5-4B", "gemma-4-E2B-it", "gemma-4-E4B-it",
        "NVIDIA-Nemotron-3-Nano-4B-BF16", "Qwen3-4B-Instruct-2507",
        "Llama-3.2-3B-Instruct",
    ]

    table_rows = []
    for model in MODEL_ORDER:
        table_rows.append({
            "name":       model,
            "is_mojo":    False,
            "cw_pos_mae": float(cat_mae.loc[POSITIVE_CATEGORIES, model].mean()),
            "cw_pos_rho": _cw_pooled_corr(cw_pairs, model, POSITIVE_CATEGORIES),
            "cw_neg_mae": float(cat_mae.loc[NEGATIVE_CATEGORIES, model].mean()),
            "cw_neg_rho": _cw_pooled_corr(cw_pairs, model, NEGATIVE_CATEGORIES),
            "swb_mae":    float(swb_cat_mae[model].mean()),
            "swb_rho":    float(swb_rho_table[model].mean()),
        })

    PAIRED = [
        ("Max (Tol=$\\infty$)", float("inf"), float("inf")),
        ("Tol=13\\%",           2.6,          0.65),
        ("Tol=11\\%",           2.2,          0.55),
        ("Tol=9\\%",            1.8,          0.45),
        ("Tol=6.5\\%",          1.3,          0.33),
    ]

    for label, cw_thr, swb_thr in PAIRED:
        pm = _mojo_mae(POSITIVE_CATEGORIES, score_table, cw_ev_names, cat_mae, cw_thr)
        nm = _mojo_mae(NEGATIVE_CATEGORIES, score_table, cw_ev_names, cat_mae, cw_thr)
        pr = _mojo_cw_pooled_corr(POSITIVE_CATEGORIES, score_table, cw_ev_names, cw_pairs, cw_thr)
        nr = _mojo_cw_pooled_corr(NEGATIVE_CATEGORIES, score_table, cw_ev_names, cw_pairs, cw_thr)
        sm, sr = _mojo_swb_corr(SWB_CATEGORIES, swb_score_table, swb_ev_names,
                                 swb_cat_mae, swb_rho_table, swb_thr)
        table_rows.append({
            "name":       f"MOJO ({label})",
            "is_mojo":    True,
            "cw_pos_mae": pm, "cw_pos_rho": pr,
            "cw_neg_mae": nm, "cw_neg_rho": nr,
            "swb_mae":    sm, "swb_rho":    sr,
        })

    # Render LaTeX
    model_only = [r for r in table_rows if not r["is_mojo"]]

    lines = [
        r"\caption{Alignment against the Claude Sonnet 4.6 baseline across both datasets."
        r" MAE~$\downarrow$, Spearman~$\rho$~$\uparrow$.}",
        r"\label{table:results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc|cc}",
        r"\toprule",
        r"& \multicolumn{4}{c|}{\textbf{Creative Writing}} & \multicolumn{2}{c}{\textbf{SWB}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}",
        r"& \multicolumn{2}{c}{\textbf{Positive}} & \multicolumn{2}{c|}{\textbf{Negative}} & & \\",
        (r"\textbf{Evaluator} & \textbf{MAE} & \textbf{$\rho$}"
         r" & \textbf{MAE} & \textbf{$\rho$}"
         r" & \textbf{MAE} & \textbf{$\rho$} \\"),
        r"\midrule",
    ]

    for r in model_only:
        n = r["name"]
        lines.append(
            f"{n:<35} & {_cell(r['cw_pos_mae'])} & {_cell(r['cw_pos_rho'])}"
            f" & {_cell(r['cw_neg_mae'])} & {_cell(r['cw_neg_rho'])}"
            f" & {_cell(r['swb_mae'])} & {_cell(r['swb_rho'])} \\\\"
        )

    lines += [
        r"\midrule",
        r"Claude Sonnet 4.6 (Ref)            & --- & --- & --- & --- & --- & --- \\",
        r"\midrule",
    ]

    for r in [r for r in table_rows if r["is_mojo"]]:
        n = r["name"]
        lines.append(
            f"{n:<35} & {_cell(r['cw_pos_mae'])} & {_cell(r['cw_pos_rho'])}"
            f" & {_cell(r['cw_neg_mae'])} & {_cell(r['cw_neg_rho'])}"
            f" & {_cell(r['swb_mae'])} & {_cell(r['swb_rho'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}%  % end resizebox",
    ]

    print("\n".join(lines))


if __name__ == "__main__":
    main()
