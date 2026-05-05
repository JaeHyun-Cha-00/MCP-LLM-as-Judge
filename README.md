# MOJO: A Mixture of Open-Weight Judges for Orchestrated LLM-as-a-Judge

Code for the NeurIPS 2026 paper: **"MOJO: A Mixture of Open-Weight Judges for Orchestrated LLM-as-a-Judge"**

---

## Overview

**Problem**: LLM-as-a-judge is powerful for evaluating subjective domains like creative writing, but using proprietary frontier models as judges is limited by high cost and rate limits.

**Approach**: MOJO statically routes each rubric to the open-weight surrogate with the highest alignment to the frontier baseline, with a configurable fallback threshold *Tol* that escalates to the frontier model when no surrogate meets the requirement. Routing calibration samples √n data points and runs exhaustive evaluation across all (rubric, evaluator) pairs; the resulting routing table is packaged as a standalone MCP server.

---

## Repository structure

```
MCP-LLM-as-Judge/
│
├── creative_writing/               # EQ-Bench Creative Writing v3 (22 rubrics, 0–20 scale)
│   ├── dataset/
│   │   ├── data.csv                # 767 AI-generated stories (text column: response)
│   │   └── results/                # Per-model evaluation CSVs  ({stem}_result.csv)
│   ├── logs/                       # Per-request LLM call logs (JSONL)
│   └── src/                        # Data-collection pipeline for this dataset
│       ├── config.py               # Endpoint configuration (OpenRouter / vLLM)
│       ├── clients.py              # OpenAI-compatible client with JSONL logging
│       ├── evaluation.py           # 22-category evaluator with JSON fallback logic
│       ├── server.py               # FastMCP server (dataset-specific batch tools)
│       └── run_baseline_evaluation.py
│
├── story_writing_benchmark/        # Story Writing Benchmark (15 rubrics, 0–5 scale)
│   ├── dataset/
│   │   ├── data.csv                # 3,480 stories (text column: story_text)
│   │   └── results/                # Per-model evaluation CSVs
│   ├── logs/
│   └── src/                        # Data-collection pipeline for this dataset
│       └── ...                     # same structure as creative_writing/src/
│
├── mojo/                           # MOJO CLI — calibration + server generation
│   ├── calibrate.py                # Stage 1 (live): call models, compute alignment
│   ├── calibrate_offline.py        # Stage 1 (offline): read result CSVs directly
│   ├── generate.py                 # Stage 2: emit a self-contained MCP server
│   ├── metrics.py                  # MAE, RMSE, Spearman, Kendall
│   ├── prompts/                    # Eval system prompts for each dataset
│   └── README.md
│
├── scripts/                        # Analysis and figure scripts (paper)
│   ├── pareto_figures.py           # Pareto operating-point curves
│   ├── calibration_robustness.py   # Seed-variance violin plots
│   └── threshold_sweep.py          # Threshold vs. cost/metric sweep plots
│
├── configs/
│   ├── model_registry.json         # alias → endpoint + result_stem for all models
│   ├── baseline.json               # Baseline endpoint config (Claude Sonnet 4.6)
│   └── models.json                 # Open-weight model endpoint configs
│
├── Makefile                        # One-command workflows (see Quick start)
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here   # only needed for live calibration / paper eval

# Calibrate from pre-computed results and generate both MCP servers
make all
```

`make all` runs offline calibration (reads the included result CSVs — no API calls) and generates two ready-to-run MCP servers:

| Output | Dataset |
|---|---|
| `creative_writing_mojo_server.py` | EQ-Bench Creative Writing v3 |
| `story_writing_mojo_server.py` | Story Writing Benchmark |

Each exposes a single MCP tool:

```python
evaluate(text: str) -> str   # returns {"scores": {"RubricName": score, ...}}
```

---

## Datasets

| Dataset | Stories | Rubrics | Scale | Source |
|---------|---------|---------|-------|--------|
| EQ-Bench Creative Writing v3 | 767 | 22 (13 positive, 9 negative) | 0–20 | `Disya/eq-bench-creative-writing-v3` |
| Story Writing Benchmark | 3,480 | 15 (all positive) | 0–5 | `lars1234/story_writing_benchmark` |

Both datasets are pre-included in `dataset/data.csv` under each module.

**Evaluated open-weight surrogates**: Qwen3-4B-Instruct-2507, Qwen3.5-4B, gemma-4-E2B-it, gemma-4-E4B-it, Llama-3.2-3B-Instruct, NVIDIA-Nemotron-3-Nano-4B-BF16 — all deployed via vLLM on an NVIDIA H100 PCIe (80 GB).

**Baseline**: Claude Sonnet 4.6 via OpenRouter.

---

## Pipeline

The repository covers three layers. Each layer is independent — you can enter at any point using the pre-computed artifacts already included.

### Layer 1 — Data collection (`creative_writing/src/`, `story_writing_benchmark/src/`)

Runs inference with a given model and writes `{name}_result.csv` to `dataset/results/`.

```bash
# Re-run baseline (Claude Sonnet 4.6 via OpenRouter)
cd creative_writing/src && python run_baseline_evaluation.py
cd story_writing_benchmark/src && python run_baseline_evaluation.py --limit 100
```

The per-dataset `src/server.py` files expose **dataset-specific batch tools** for use inside Claude (configured via `.mcp.json`):

| Tool | Description |
|------|-------------|
| `evaluate_single_story(story)` | Evaluate one story across all 22 CW rubrics |
| `evaluate_full_dataset(output_filename)` | Evaluate full CW dataset, save to CSV |
| `swb_evaluate_single_story(story)` | Evaluate one story across all 15 SWB rubrics |
| `swb_evaluate_full_dataset(output_filename)` | Evaluate full SWB dataset, save to CSV |

These tools call a single designated model for all rubrics. They are used to produce the result CSVs, not for the routed evaluation.

### Layer 2 — Calibration + server generation (`mojo/`)

Reads the result CSVs, computes per-rubric alignment, and generates a routed MCP server where each rubric is dispatched to its best-aligned surrogate.

```bash
# Offline (uses included CSVs)
make calibrate-cw          # → creative_writing_mojo_config.json
make calibrate-sw          # → story_writing_mojo_config.json
make generate-cw           # → creative_writing_mojo_server.py
make generate-sw           # → story_writing_mojo_server.py

# Live (calls models via API, re-samples √n rows)
make calibrate-cw-live
make calibrate-sw-live

# Change metric or threshold
make calibrate-cw METRIC_CW=spearman TOL_CW=0.5
```

The generated server's `evaluate(text)` tool batches rubrics by assigned model and calls them concurrently. Rubrics where no surrogate clears *Tol* fall back to the baseline endpoint.

### Layer 3 — Analysis (`scripts/`)

Reproduces the figures in the paper. All scripts read from the pre-computed result CSVs; no model calls required.

```bash
python scripts/pareto_figures.py          # Pareto operating-point curves
python scripts/calibration_robustness.py  # Seed-variance violin plots (Fig. 3)
python scripts/threshold_sweep.py         # Threshold sweep (Fig. 4)
```

Figures are written to `scripts/figures/`.

---

## Setup details

### API key

Both the data-collection layer and live calibration read `OPENROUTER_API_KEY` from the environment. The per-dataset `src/` scripts additionally support a `.env` file via `python-dotenv`:

```bash
cp .env.example .env   # then edit OPENROUTER_API_KEY=...
```

The `mojo/` CLI and the generated servers read the key directly from the environment — no `.env` loading.

### MCP integration (Claude Desktop / Claude Code)

```bash
cp .mcp.json.example .mcp.json
# Edit: replace YOUR_PROJECT_PATH with the absolute path to this repo
```

This wires the **dataset-specific batch tools** (Layer 1) into Claude. To use the **MOJO-generated routed server** instead, point `.mcp.json` at the generated `*_mojo_server.py`.

### Local vLLM backend

To run the surrogates locally, update `src/config.py` in either dataset module:

```python
base_url: str = "http://localhost:8001/v1"
model: str = "Qwen3-4B-Instruct-2507"
```

The `configs/model_registry.json` maps each model alias to its default port assignment (8001–8006).
