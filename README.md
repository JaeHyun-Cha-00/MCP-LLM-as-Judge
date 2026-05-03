# MOJO: A Mixture of Open-Weight Judges for Orchestrated LLM-as-a-Judge

Code for the NeurIPS 2026: **"MOJO: A Mixture of Open-Weight Judges for Orchestrated LLM-as-a-Judge"**

---

## Overview

**Problem**: LLM-as-a-judge is powerful for evaluating subjective domains like creative writing, but using proprietary frontier models as judges is limited by high cost and rate limits.

**MOJO** (Mixture-Of-Judges Orchestration) addresses this by routing evaluation tasks to cost-effective open-weight alternatives via the **Model Context Protocol (MCP)**. MOJO statically routes each rubric to the open-weight model with the highest alignment to the frontier judge, with a configurable fallback threshold Tol that escalates to the frontier model when no surrogate meets the alignment requirement.

MOJO works in three phases:
1. **Candidate Selection** — Filter open-weight models using MAE/RMSE to remove those with instruction-following failures or output collapse.
2. **Routing Calibration** — Sample √n data points and run exhaustive evaluation across all (rubric, evaluator) pairs to determine the best surrogate per rubric.
3. **Evaluation** — Route each rubric to its assigned model via the MCP server; fall back to the frontier model for rubrics where no surrogate clears threshold Tol.

---

## Repository Structure

```
MCP-LLM-as-Judge/
├── creative_writing/               # EQ-Bench Creative Writing v3 (22 rubrics, 0–20 scale)
│   ├── dataset/
│   │   ├── data.csv                # 767 AI-generated stories
│   │   └── results/                # Per-model evaluation results (CSV)
│   ├── logs/                       # Per-request LLM call logs (JSONL)
│   ├── notebooks/
│   │   └── analysis.ipynb          # Alignment analysis (MAE, RMSE, Spearman ρ, Kendall τ)
│   └── src/
│       ├── config.py               # Endpoint configuration (OpenRouter / vLLM)
│       ├── clients.py              # OpenAI-compatible client with JSONL logging
│       ├── evaluation.py           # 22-category evaluator with JSON fallback logic
│       ├── server.py               # FastMCP server exposing MCP tools
│       └── run_baseline_evaluation.py  # Baseline (Claude Sonnet 4.6) evaluation script
│
├── story_writing_benchmark/        # Story Writing Benchmark (15 rubrics, 0–5 scale)
│   ├── dataset/
│   │   ├── data.csv                # 3,480 English stories (lars1234/story_writing_benchmark)
│   │   └── results/                # Per-model evaluation results (CSV)
│   ├── logs/                       # Per-request LLM call logs (JSONL)
│   ├── notebooks/
│   │   └── analysis.ipynb          # Alignment analysis
│   └── src/
│       ├── config.py               # Endpoint configuration (OpenRouter / vLLM)
│       ├── clients.py              # OpenAI-compatible client with JSONL logging
│       ├── evaluation.py           # 15-category evaluator with JSON fallback logic
│       ├── server.py               # FastMCP server exposing MCP tools
│       └── run_baseline_evaluation.py  # Baseline (Claude Sonnet 4.6) evaluation script
│
├── .env.example                    # API key template
├── .mcp.json.example               # MCP server config template
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Configure MCP servers (for MCP tool usage with Claude)

```bash
cp .mcp.json.example .mcp.json
```

Edit `.mcp.json` and replace `YOUR_PROJECT_PATH` with the absolute path to this repository.

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

## Usage

### MCP tools (via Claude)

With `.mcp.json` configured, the following tools are available in Claude:

| Tool | Description |
|------|-------------|
| `evaluate_single_story(story)` | Evaluate one story across all 22 CW rubrics |
| `evaluate_full_dataset(output_filename)` | Evaluate full CW dataset, save to CSV |
| `swb_evaluate_single_story(story)` | Evaluate one story across all 15 SWB rubrics |
| `swb_evaluate_full_dataset(output_filename)` | Evaluate full SWB dataset, save to CSV |

### Reproducing baseline evaluation

To re-run the Claude Sonnet 4.6 baseline via OpenRouter:

```bash
# Creative Writing
cd creative_writing/src
python run_baseline_evaluation.py

# Story Writing Benchmark (full dataset)
cd story_writing_benchmark/src
python run_baseline_evaluation.py

# Story Writing Benchmark (first N entries)
python run_baseline_evaluation.py --limit 100
```

Results are saved to `dataset/results/`.

### Using a local vLLM backend

To evaluate open-weight models, update `src/config.py` in either module:

```python
# Uncomment and configure:
base_url: str = "http://localhost:8001/v1"  # your vLLM server address
model: str = "meta-llama/Llama-3.2-3B-Instruct"  # model served by vLLM
```

---

## Pre-computed Results

All evaluation results used in the paper are included in `dataset/results/` under each module. These can be used to reproduce the paper's analysis without re-running inference.