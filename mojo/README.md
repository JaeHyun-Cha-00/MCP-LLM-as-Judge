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

## Two ways to calibrate

| | `calibrate` | `calibrate-offline` |
|---|---|---|
| **Data source** | Calls models live via API | Reads pre-computed `{stem}_result.csv` files |
| **Model config** | `--baseline-model` + `--open-weight-models` | `--model-registry` (contains both stems and endpoints) |
| **Sample size** | √n rows sampled at runtime | All rows in the CSVs |
| **Output** | Same `mojo_config.json` | Same `mojo_config.json` |

`generate` and the generated server are identical regardless of which calibration path produced the config.

---

## Examples

Both examples below share the same baseline and open-weight model files.

**`baseline.json`** — claude-sonnet-4-6 via OpenRouter (the reference judge):
```json
{
  "base_url":    "https://openrouter.ai/api/v1",
  "model_id":    "anthropic/claude-sonnet-4-6",
  "api_key_env": "OPENROUTER_API_KEY",
  "temperature": 0.0,
  "max_tokens":  2048
}
```

**`models.json`** — six small open-weight models served locally via vLLM:
```json
{
  "qwen3-4b": {
    "base_url": "http://localhost:8001/v1", "model_id": "Qwen3-4B-Instruct-2507",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  },
  "qwen3.5-4b": {
    "base_url": "http://localhost:8002/v1", "model_id": "Qwen3.5-4B",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  },
  "gemma-4-2b": {
    "base_url": "http://localhost:8003/v1", "model_id": "gemma-4-E2B-it",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  },
  "gemma-4-4b": {
    "base_url": "http://localhost:8004/v1", "model_id": "gemma-4-E4B-it",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  },
  "llama-3.2-3b": {
    "base_url": "http://localhost:8005/v1", "model_id": "Llama-3.2-3B-Instruct",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  },
  "nemotron-3-4b": {
    "base_url": "http://localhost:8006/v1", "model_id": "NVIDIA-Nemotron-3-Nano-4B-BF16",
    "api_key_env": "EMPTY", "temperature": 0.0, "max_tokens": 1024
  }
}
```

---

### Example 1 — EQ-Bench Creative Writing (767 stories, 22 rubrics, 0–20 scale)

Dataset: `creative_writing/dataset/data.csv`, text column: `response`, sample size: √767 ≈ 27.

The calibration prompt matches the scoring scale (0–20 float) used by the baseline.

**Calibrate:**

```bash
python -m mojo calibrate \
  --dataset            creative_writing/dataset/data.csv \
  --text-column        response \
  --eval-prompt        "You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and numeric scores (0.0-20.0) as values. Remember: positive metrics should have higher scores, negative/penalty metrics should have lower scores." \
  --rubrics \
    "Adherence to Instructions" \
    "Believable Character Actions" \
    "Nuanced Characters" \
    "Consistent Voice / Tone of Writing" \
    "Imagery and Descriptive Quality" \
    "Elegant Prose" \
    "Emotionally Engaging" \
    "Emotionally Complex" \
    "Coherent" \
    "Well-earned Lightness or Darkness" \
    "Sentences Flow Naturally" \
    "Overall Reader Engagement" \
    "Overall Impression" \
    "Meandering" \
    "Weak Dialogue" \
    "Tell-Don't-Show" \
    "Unsurprising or Uncreative" \
    "Amateurish" \
    "Purple Prose" \
    "Overwrought" \
    "Incongruent Ending Positivity" \
    "Unearned Transformations" \
  --baseline-model     baseline.json \
  --open-weight-models models.json \
  --metric             mae \
  --tolerance          2.0 \
  --output             creative_writing_mojo_config.json
```

**Generate:**

```bash
python -m mojo generate \
  --config  creative_writing_mojo_config.json \
  --output  creative_writing_mojo_server.py
```

**Run:**

```bash
python creative_writing_mojo_server.py
```

The generated server exposes `evaluate(text)` and returns scores for all 22 rubrics, each graded by whichever model had the lowest MAE against claude-sonnet-4-6 on the calibration sample. Any rubric where no model stayed within MAE ≤ 2.0 falls back to the baseline.

---

### Example 2 — lars1234/story_writing_benchmark (3 480 stories, 15 rubrics, 0–5 scale)

Dataset: `story_writing_benchmark/dataset/data.csv`, text column: `story_text`, sample size: √3480 ≈ 59.

Spearman correlation is used here because the rubrics are integer-scored (0–5) and ordinal alignment matters more than absolute error.

**Calibrate:**

```bash
python -m mojo calibrate \
  --dataset            story_writing_benchmark/dataset/data.csv \
  --text-column        story_text \
  --eval-prompt        "You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and integer scores (0-5) as values." \
  --rubrics \
    "Grammar, Spelling, and Punctuation Quality" \
    "Clarity and Understandability" \
    "Logical Connection Between Events and Ideas" \
    "Scene Construction and Purpose" \
    "Internal Consistency" \
    "Character Consistency" \
    "Character Motivation and Actions" \
    "Sentence Pattern Variety" \
    "Avoidance of Clichés and Overused Phrases" \
    "Natural Dialogue" \
    "Avoidance of Predictable Narrative Tropes" \
    "Character Depth and Dimensionality" \
    "Realistic Character Interactions" \
    "Ability to Hold Reader Interest" \
    "Satisfying Plot Resolution" \
  --baseline-model     baseline.json \
  --open-weight-models models.json \
  --metric             spearman \
  --tolerance          0.6 \
  --output             story_writing_mojo_config.json
```

**Generate:**

```bash
python -m mojo generate \
  --config  story_writing_mojo_config.json \
  --output  story_writing_mojo_server.py
```

**Run:**

```bash
python story_writing_mojo_server.py
```

Any rubric where no open-weight model reached Spearman ρ ≥ 0.6 against the baseline falls back to claude-sonnet-4-6.

---

### Example 1 (offline) — EQ-Bench Creative Writing

```bash
python -m mojo calibrate-offline \
  --results-dir    creative_writing/dataset/results \
  --baseline-stem  claude_sonnet_4.6 \
  --model-registry configs/model_registry.json \
  --eval-prompt    "You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and numeric scores (0.0-20.0) as values. Remember: positive metrics should have higher scores, negative/penalty metrics should have lower scores." \
  --rubrics \
    "Adherence to Instructions" \
    "Believable Character Actions" \
    "Nuanced Characters" \
    "Consistent Voice / Tone of Writing" \
    "Imagery and Descriptive Quality" \
    "Elegant Prose" \
    "Emotionally Engaging" \
    "Emotionally Complex" \
    "Coherent" \
    "Well-earned Lightness or Darkness" \
    "Sentences Flow Naturally" \
    "Overall Reader Engagement" \
    "Overall Impression" \
    "Meandering" \
    "Weak Dialogue" \
    "Tell-Don't-Show" \
    "Unsurprising or Uncreative" \
    "Amateurish" \
    "Purple Prose" \
    "Overwrought" \
    "Incongruent Ending Positivity" \
    "Unearned Transformations" \
  --metric    mae \
  --tolerance 2.0 \
  --output    creative_writing_mojo_config.json
```

### Example 2 (offline) — story_writing_benchmark

```bash
python -m mojo calibrate-offline \
  --results-dir    story_writing_benchmark/dataset/results \
  --baseline-stem  claude_sonnet_4.6 \
  --model-registry configs/model_registry.json \
  --eval-prompt    "You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and integer scores (0-5) as values." \
  --rubrics \
    "Grammar, Spelling, and Punctuation Quality" \
    "Clarity and Understandability" \
    "Logical Connection Between Events and Ideas" \
    "Scene Construction and Purpose" \
    "Internal Consistency" \
    "Character Consistency" \
    "Character Motivation and Actions" \
    "Sentence Pattern Variety" \
    "Avoidance of Clichés and Overused Phrases" \
    "Natural Dialogue" \
    "Avoidance of Predictable Narrative Tropes" \
    "Character Depth and Dimensionality" \
    "Realistic Character Interactions" \
    "Ability to Hold Reader Interest" \
    "Satisfying Plot Resolution" \
  --metric    spearman \
  --tolerance 0.6 \
  --output    story_writing_mojo_config.json
```

Then generate the server from either config:

```bash
python -m mojo generate --config creative_writing_mojo_config.json --output creative_writing_mojo_server.py
python -m mojo generate --config story_writing_mojo_config.json    --output story_writing_mojo_server.py
```

---

## Usage (generic)

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
