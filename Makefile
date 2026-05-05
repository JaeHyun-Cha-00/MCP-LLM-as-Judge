# ==============================================================================
# MOJO — Mixture of Open-weight Orchestrator
#
# Offline calibration (default): reads pre-computed result CSVs, free to run.
# Live calibration:              calls models via API, requires running servers.
#
# Targets
#   make all                  offline calibrate + generate for both datasets
#   make calibrate-cw         offline calibrate → creative_writing_mojo_config.json
#   make calibrate-sw         offline calibrate → story_writing_mojo_config.json
#   make generate-cw          generate → creative_writing_mojo_server.py
#   make generate-sw          generate → story_writing_mojo_server.py
#   make calibrate-cw-live    live calibrate (always runs, overwrites config)
#   make calibrate-sw-live    live calibrate (always runs, overwrites config)
#   make clean                remove generated configs and servers
#
# Override defaults, e.g.:
#   make calibrate-cw METRIC_CW=spearman TOL_CW=0.5
# ==============================================================================

# ------------------------------------------------------------------------------
# Tunable knobs
# ------------------------------------------------------------------------------
METRIC_CW ?= mae
METRIC_SW ?= mae
TOL_CW    ?= 20.0
TOL_SW    ?= 5.0

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
REGISTRY    = configs/model_registry.json
BASELINE    = configs/baseline.json
MODELS      = configs/models.json

PROMPT_CW   = @mojo/prompts/creative_writing_eval.txt
PROMPT_SW   = @mojo/prompts/story_writing_eval.txt

RESULTS_CW  = creative_writing/dataset/results
RESULTS_SW  = story_writing_benchmark/dataset/results

DATASET_CW  = creative_writing/dataset/data.csv
DATASET_SW  = story_writing_benchmark/dataset/data.csv

CONFIG_CW   = creative_writing_mojo_config.json
CONFIG_SW   = story_writing_mojo_config.json
SERVER_CW   = creative_writing_mojo_server.py
SERVER_SW   = story_writing_mojo_server.py

# ------------------------------------------------------------------------------
# Rubrics
# ------------------------------------------------------------------------------
RUBRICS_CW = \
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
	"Unearned Transformations"

RUBRICS_SW = \
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
	"Satisfying Plot Resolution"

# ==============================================================================
# Top-level
# ==============================================================================

.PHONY: all clean \
        calibrate-cw calibrate-sw \
        calibrate-cw-live calibrate-sw-live \
        generate-cw generate-sw

all: $(SERVER_CW) $(SERVER_SW)

# Phony aliases so you can type `make calibrate-cw` instead of the filename
calibrate-cw: $(CONFIG_CW)
calibrate-sw: $(CONFIG_SW)
generate-cw:  $(SERVER_CW)
generate-sw:  $(SERVER_SW)

# ==============================================================================
# Stage 1 — Offline calibration (reads pre-computed CSVs)
# Skips if the config file already exists; run `make clean` to force a re-run.
# ==============================================================================

$(CONFIG_CW):
	python -m mojo calibrate-offline \
		--results-dir    $(RESULTS_CW) \
		--baseline-stem  claude_sonnet_4.6 \
		--model-registry $(REGISTRY) \
		--eval-prompt    $(PROMPT_CW) \
		--rubrics        $(RUBRICS_CW) \
		--metric         $(METRIC_CW) \
		--tolerance      $(TOL_CW) \
		--output         $@

$(CONFIG_SW):
	python -m mojo calibrate-offline \
		--results-dir    $(RESULTS_SW) \
		--baseline-stem  claude_sonnet_4.6 \
		--model-registry $(REGISTRY) \
		--eval-prompt    $(PROMPT_SW) \
		--rubrics        $(RUBRICS_SW) \
		--metric         $(METRIC_SW) \
		--tolerance      $(TOL_SW) \
		--output         $@

# ==============================================================================
# Stage 1 — Live calibration (calls models via API, always re-runs)
# Requires: OPENROUTER_API_KEY set, local vLLM servers on ports 8001-8006.
# ==============================================================================

calibrate-cw-live:
	python -m mojo calibrate \
		--dataset            $(DATASET_CW) \
		--text-column        response \
		--eval-prompt        $(PROMPT_CW) \
		--rubrics            $(RUBRICS_CW) \
		--baseline-model     $(BASELINE) \
		--open-weight-models $(MODELS) \
		--metric             $(METRIC_CW) \
		--tolerance          $(TOL_CW) \
		--output             $(CONFIG_CW)

calibrate-sw-live:
	python -m mojo calibrate \
		--dataset            $(DATASET_SW) \
		--text-column        story_text \
		--eval-prompt        $(PROMPT_SW) \
		--rubrics            $(RUBRICS_SW) \
		--baseline-model     $(BASELINE) \
		--open-weight-models $(MODELS) \
		--metric             $(METRIC_SW) \
		--tolerance          $(TOL_SW) \
		--output             $(CONFIG_SW)

# ==============================================================================
# Stage 2 — Generate MCP server from config
# ==============================================================================

$(SERVER_CW): $(CONFIG_CW)
	python -m mojo generate \
		--config  $< \
		--output  $@

$(SERVER_SW): $(CONFIG_SW)
	python -m mojo generate \
		--config  $< \
		--output  $@

# ==============================================================================
# Notebooks (executed outputs only, no metadata)
# ==============================================================================

.PHONY: notebooks-cw notebooks-sw notebooks

notebooks-cw:
	jupyter nbconvert --execute --inplace creative_writing/notebooks/*.ipynb
	python scripts/clean_notebook_metadata.py creative_writing/notebooks/*.ipynb

notebooks-sw:
	jupyter nbconvert --execute --inplace story_writing_benchmark/notebooks/*.ipynb
	python scripts/clean_notebook_metadata.py story_writing_benchmark/notebooks/*.ipynb

notebooks: notebooks-cw notebooks-sw

# ==============================================================================
# Figure generation (reproduces paper figures from pre-computed CSVs)
# ==============================================================================

.PHONY: figures figures-pareto figures-robustness figures-sweep

figures: figures-pareto figures-robustness figures-sweep

figures-pareto:
	python scripts/pareto_figures.py

figures-robustness:
	python scripts/calibration_robustness.py

figures-sweep:
	python scripts/threshold_sweep.py

# ==============================================================================
# Housekeeping
# ==============================================================================

clean:
	rm -f $(CONFIG_CW) $(CONFIG_SW) $(SERVER_CW) $(SERVER_SW)
