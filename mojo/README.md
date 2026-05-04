# MOJO — Mixture of Open-weight Orchestrator

MOJO is a two-stage tool that routes each rubric in an LLM evaluation to the open-weight model that best aligns with a proprietary baseline, then packages the result as a runnable MCP server.

```
Dataset + rubrics + models
        │
        ▼
┌───────────────┐   mojo_config.json   ┌─────────────────┐
│  calibrate    │ ──────────────────▶  │    generate     │
│  (Stage 1)    │                      │   (Stage 2)     │
└───────────────┘                      └─────────────────┘
                                               │
                                               ▼
                                        mojo_server.py
                                    (runnable MCP server)
```

---

## How it works

### Stage 1 — Calibrate

1. Samples **√n** rows from the dataset at random (reproducible seed).
2. Calls the **baseline model** and every **open-weight model** on each sample, asking for a JSON score for every rubric.
3. For each rubric, computes an alignment metric between every open-weight model and the baseline.
4. Selects the best-aligned model per rubric. If a tolerance threshold is set and no model meets it, the rubric falls back to the baseline.
5. Writes a `mojo_config.json` that records the routing decision for every rubric.

### Stage 2 — Generate

Reads `mojo_config.json` and writes a self-contained `mojo_server.py`. The generated server:

- Exposes a single MCP tool `evaluate(text: str) -> str`.
- Groups rubrics by their assigned model and issues one API call per model concurrently.
- Returns `{"scores": {"RubricName": score, ...}}` as JSON.

---

## Inputs

| Argument | Description |
|---|---|
| `--dataset` | CSV file with at least one text column (default column name: `text`). |
| `--eval-prompt` | System prompt for the judge LLM. Prefix with `@` to read from a file. |
| `--rubrics` | One or more rubric names (space-separated). |
| `--baseline-model` | Baseline endpoint config — JSON string or path to a `.json` file. |
| `--open-weight-models` | Open-weight model configs — JSON dict `{"alias": endpoint_cfg, ...}` or path to a `.json` file. |
| `--metric` | Alignment metric: `mae`, `rmse`, `spearman`, or `kendall`. |
| `--tolerance` | *(optional)* Fallback threshold. For `mae`/`rmse`: maximum tolerable error. For `spearman`/`kendall`: minimum acceptable correlation. If omitted, the best open-weight model always wins. |
| `--text-column` | *(optional)* Column name for the text in the dataset (default: `text`). |
| `--output` | *(optional)* Output path for the config (default: `mojo_config.json`). |

### Endpoint config format

Both `--baseline-model` and the values inside `--open-weight-models` use the same shape:

```json
{
  "base_url":    "https://openrouter.ai/api/v1",
  "model_id":    "anthropic/claude-sonnet-4-6",
  "api_key_env": "OPENROUTER_API_KEY",
  "temperature": 0.0,
  "max_tokens":  2048
}
```

`api_key_env` names an environment variable that holds the API key. Set it to `"EMPTY"` for local servers that need no authentication.

### Alignment metrics

| Metric | Better when | Tolerance fallback when |
|---|---|---|
| `mae` | lower | `best > tolerance` |
| `rmse` | lower | `best > tolerance` |
| `spearman` | higher | `best < tolerance` |
| `kendall` | higher | `best < tolerance` |

---

## Usage

### Inline JSON models

```bash
python -m mojo calibrate \
  --dataset       creative_writing/dataset/data.csv \
  --text-column   story \
  --eval-prompt   "You are an expert literary critic." \
  --rubrics       "Clarity" "Coherence" "Elegant Prose" \
  --baseline-model \
    '{"base_url":"https://openrouter.ai/api/v1","model_id":"anthropic/claude-sonnet-4-6","api_key_env":"OPENROUTER_API_KEY","temperature":0.0,"max_tokens":2048}' \
  --open-weight-models \
    '{"qwen3-4b":{"base_url":"http://localhost:8001/v1","model_id":"Qwen3-4B","api_key_env":"EMPTY","temperature":0.0,"max_tokens":1024},"llama-3b":{"base_url":"http://localhost:8002/v1","model_id":"Llama-3.2-3B","api_key_env":"EMPTY","temperature":0.0,"max_tokens":1024}}' \
  --metric        mae \
  --tolerance     2.0 \
  --output        mojo_config.json
```

### Models from a file

Put your open-weight model configs in `models.json`:

```json
{
  "qwen3-4b": {
    "base_url":    "http://localhost:8001/v1",
    "model_id":    "Qwen3-4B",
    "api_key_env": "EMPTY",
    "temperature": 0.0,
    "max_tokens":  1024
  },
  "llama-3b": {
    "base_url":    "http://localhost:8002/v1",
    "model_id":    "Llama-3.2-3B-Instruct",
    "api_key_env": "EMPTY",
    "temperature": 0.0,
    "max_tokens":  1024
  }
}
```

Then:

```bash
python -m mojo calibrate \
  --dataset             data.csv \
  --eval-prompt         @prompts/eval.txt \
  --rubrics             "Grammar" "Clarity" "Engagement" \
  --baseline-model      baseline.json \
  --open-weight-models  models.json \
  --metric              spearman \
  --output              mojo_config.json
```

### Generate the MCP server

```bash
python -m mojo generate \
  --config  mojo_config.json \
  --output  mojo_server.py
```

### Run the generated server

```bash
# stdio (default — for MCP clients like Claude Desktop)
python mojo_server.py

# SSE
python mojo_server.py   # add transport args if you patch the __main__ block
```

---

## Config file format

`mojo_config.json` is the contract between Stage 1 and Stage 2.

```json
{
  "server_name": "mojo-judge",
  "eval_prompt": "You are an expert literary critic.",
  "metric": "mae",
  "tolerance": 2.0,
  "baseline_endpoint": { "..." : "..." },
  "open_weight_models": { "alias": { "..." : "..." } },
  "rubric_routing": [
    {
      "rubric":       "Clarity",
      "best_model":   "qwen3-4b",
      "metric":       "mae",
      "metric_value": 1.23,
      "use_baseline": false,
      "endpoint":     { "..." : "..." }
    },
    {
      "rubric":       "Elegance",
      "best_model":   "llama-3b",
      "metric":       "mae",
      "metric_value": 3.10,
      "use_baseline": true,
      "endpoint":     { "baseline endpoint — tolerance exceeded" : "" }
    }
  ]
}
```

`use_baseline: true` means the open-weight model's metric value exceeded the tolerance and the rubric fell back to the baseline endpoint.

---

## Generated server

`mojo_server.py` is a standalone FastMCP server with no dependency on the MOJO package. It requires only `fastmcp` and `openai`.

The single tool it exposes:

```
evaluate(text: str) -> str
```

Returns:

```json
{"scores": {"Clarity": 7.5, "Coherence": 8.0, "Elegant Prose": 6.5}}
```

Rubrics assigned to the same model are batched into one API call. Rubrics that fell back to the baseline are also batched with any other baseline-routed rubrics.

---

## File structure

```
mojo/
├── __init__.py
├── __main__.py   — CLI (calibrate / generate subcommands)
├── calibrate.py  — Stage 1: sampling, evaluation, metric computation, config output
├── generate.py   — Stage 2: MCP server code generation
├── metrics.py    — MAE, RMSE, Spearman, Kendall + best-model selection
└── README.md     — this file
```
