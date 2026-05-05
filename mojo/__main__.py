"""MOJO — Mixture of Open-weight Orchestrator.

Two-stage CLI:
  python -m mojo calibrate  ...   # sample dataset, evaluate models, emit config
  python -m mojo generate   ...   # turn config into a runnable MCP server

Model config format (JSON string or path to .json file):
  Baseline:  {"base_url": "...", "model_id": "...", "api_key_env": "...",
               "temperature": 0.0, "max_tokens": 2048}
  Models:    {"alias1": {same keys}, "alias2": {...}}

Example
-------
python -m mojo calibrate \\
    --dataset       data.csv \\
    --eval-prompt   "You are an expert literary critic." \\
    --rubrics       "Clarity" "Coherence" "Elegance" \\
    --baseline-model  '{"base_url":"https://openrouter.ai/api/v1","model_id":"anthropic/claude-sonnet-4-6","api_key_env":"OPENROUTER_API_KEY","temperature":0.0,"max_tokens":2048}' \\
    --open-weight-models models.json \\
    --metric        mae \\
    --tolerance     2.0 \\
    --output        mojo_config.json

python -m mojo generate \\
    --config  mojo_config.json \\
    --output  mojo_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_arg(value: str) -> dict:
    """Accept a JSON string or a path to a .json file."""
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text())
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        sys.exit(f"[mojo] Cannot parse as JSON and file not found: {value!r}\n{exc}")


def _load_prompt(value: str) -> str:
    """Accept a literal string or @path/to/file.txt."""
    if value.startswith("@"):
        p = Path(value[1:])
        if not p.exists():
            sys.exit(f"[mojo] Prompt file not found: {p}")
        return p.read_text()
    return value


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibrate import run

    eval_prompt  = _load_prompt(args.eval_prompt)
    baseline_cfg = _load_json_arg(args.baseline_model)
    model_cfgs   = _load_json_arg(args.open_weight_models)

    asyncio.run(run(
        dataset_path  = args.dataset,
        text_column   = args.text_column,
        eval_prompt   = eval_prompt,
        rubrics       = args.rubrics,
        baseline_cfg  = baseline_cfg,
        model_cfgs    = model_cfgs,
        metric        = args.metric,
        tolerance     = args.tolerance,
        output_path   = args.output,
    ))


def cmd_calibrate_offline(args: argparse.Namespace) -> None:
    from .calibrate_offline import run

    registry = _load_json_arg(args.model_registry)
    registry = {k: v for k, v in registry.items() if not k.startswith("_")}
    eval_prompt = _load_prompt(args.eval_prompt)

    run(
        results_dir   = args.results_dir,
        baseline_stem = args.baseline_stem,
        model_registry = registry,
        eval_prompt   = eval_prompt,
        rubrics       = args.rubrics,
        metric        = args.metric,
        tolerance     = args.tolerance,
        output_path   = args.output,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    from .generate import render
    render(args.config, args.output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mojo",
        description="MOJO: Mixture of Open-weight Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- calibrate ------------------------------------------------------------
    cal = sub.add_parser(
        "calibrate",
        help="Calibrate per-rubric model routing from a dataset.",
    )
    cal.add_argument(
        "--dataset", required=True, metavar="PATH",
        help="CSV file with at least a text column.",
    )
    cal.add_argument(
        "--eval-prompt", required=True, metavar="TEXT",
        help="System prompt for the judge LLM. Prefix with @ to read from file.",
    )
    cal.add_argument(
        "--rubrics", required=True, nargs="+", metavar="RUBRIC",
        help="Rubric names to evaluate (space-separated).",
    )
    cal.add_argument(
        "--baseline-model", required=True, metavar="JSON",
        help="Baseline endpoint config: JSON string or path to .json file.",
    )
    cal.add_argument(
        "--open-weight-models", required=True, metavar="JSON",
        help='Open-weight models: JSON dict {"alias": endpoint_cfg, ...} or path to .json file.',
    )
    cal.add_argument(
        "--metric", required=True,
        choices=["mae", "rmse", "spearman", "kendall"],
        help="Alignment metric used to compare open-weight models against the baseline.",
    )
    cal.add_argument(
        "--tolerance", type=float, default=None, metavar="FLOAT",
        help=(
            "Fallback threshold. For mae/rmse: max tolerable error. "
            "For spearman/kendall: min acceptable correlation. "
            "Omit to never fall back to the baseline."
        ),
    )
    cal.add_argument(
        "--text-column", default="text", metavar="COL",
        help="Name of the text column in the dataset (default: text).",
    )
    cal.add_argument(
        "--output", default="mojo_config.json", metavar="PATH",
        help="Output config path (default: mojo_config.json).",
    )
    cal.set_defaults(func=cmd_calibrate)

    # -- calibrate-offline ----------------------------------------------------
    off = sub.add_parser(
        "calibrate-offline",
        help="Calibrate from pre-computed result CSVs instead of calling models live.",
    )
    off.add_argument(
        "--results-dir", required=True, metavar="PATH",
        help="Directory containing {stem}_result.csv files.",
    )
    off.add_argument(
        "--baseline-stem", required=True, metavar="STEM",
        help="result_stem of the baseline model (e.g. claude_sonnet_4.6).",
    )
    off.add_argument(
        "--model-registry", required=True, metavar="JSON",
        help="Model registry: JSON dict or path to .json file mapping alias → endpoint + result_stem.",
    )
    off.add_argument(
        "--eval-prompt", required=True, metavar="TEXT",
        help="System prompt embedded in the generated server. Prefix with @ to read from file.",
    )
    off.add_argument(
        "--rubrics", required=True, nargs="+", metavar="RUBRIC",
        help="Rubric names (must match {rubric}_score columns in the result CSVs).",
    )
    off.add_argument(
        "--metric", required=True,
        choices=["mae", "rmse", "spearman", "kendall"],
        help="Alignment metric.",
    )
    off.add_argument(
        "--tolerance", type=float, default=None, metavar="FLOAT",
        help="Fallback threshold (same semantics as calibrate).",
    )
    off.add_argument(
        "--output", default="mojo_config.json", metavar="PATH",
        help="Output config path (default: mojo_config.json).",
    )
    off.set_defaults(func=cmd_calibrate_offline)

    # -- generate -------------------------------------------------------------
    gen = sub.add_parser(
        "generate",
        help="Generate a runnable MCP server from a MOJO calibration config.",
    )
    gen.add_argument(
        "--config", default="mojo_config.json", metavar="PATH",
        help="MOJO config produced by calibrate (default: mojo_config.json).",
    )
    gen.add_argument(
        "--output", default="mojo_server.py", metavar="PATH",
        help="Output MCP server file (default: mojo_server.py).",
    )
    gen.set_defaults(func=cmd_generate)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
