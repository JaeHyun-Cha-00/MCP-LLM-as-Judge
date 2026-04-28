from fastmcp import FastMCP
from evaluation import StoryEvaluator
from clients import WolverineClient
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

mcp = FastMCP(name="MCP-LLM Judge Story Writing Benchmark")
client = WolverineClient()
evaluator = StoryEvaluator(client)

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"

# Reference score columns from lars1234/story_writing_benchmark
REFERENCE_COLS = [f"q{i}" for i in range(1, 16)]

_dataset = None


def load_dataset():
    global _dataset
    if _dataset is None:
        path_str = str(DATASET_PATH)
        if os.path.exists(path_str):
            try:
                _dataset = pd.read_csv(path_str)
                print(f"[INFO] Dataset loaded: {len(_dataset)} entries from {path_str}")
            except Exception as e:
                print(f"[ERROR] Failed to load dataset: {e}")
                _dataset = pd.DataFrame()
        else:
            print(f"[WARNING] Dataset file not found: {path_str}")
            print(f"[INFO] Run story_writing_benchmark/download_dataset.py to download the dataset first.")
            _dataset = pd.DataFrame()
    return _dataset


def ensure_results_dir():
    try:
        os.makedirs(str(RESULTS_DIR), exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create results directory: {e}")


ensure_results_dir()


@mcp.tool()
def swb_evaluate_single_story(story: str) -> dict[str, dict]:
    """[Story Writing Benchmark] Evaluate a single story across all evaluation categories (for testing purposes)."""
    print("[INFO] Tool called: evaluate_single_story")
    results = evaluator.evaluate_all_categories(story)
    return {cat: res.to_dict() for cat, res in results.items()}


@mcp.tool()
def swb_evaluate_full_dataset(output_filename: str = None) -> dict:
    """[Story Writing Benchmark] Evaluate the entire story writing benchmark dataset and save results to CSV."""
    print("[INFO] Tool called: evaluate_full_dataset")
    dataset = load_dataset()

    if dataset.empty:
        return {"error": "Dataset is not loaded or is empty"}

    total_entries = len(dataset)

    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_results_full_{total_entries}_{timestamp}.csv"

    ensure_results_dir()
    output_path = str(RESULTS_DIR / output_filename)

    results = []
    for i in range(total_entries):
        print(f"[INFO] Evaluating entry {i + 1}/{total_entries}")
        row = dataset.iloc[i]
        story = str(row.get("story_text", ""))
        model = str(row.get("model_name", ""))

        eval_results = evaluator.evaluate_all_categories(story)

        result_row = {
            "index": i,
            "prompt_id": str(row.get("prompt_id", "")),
            "model": model,
            "language": str(row.get("language", "")),
            "theme": str(row.get("theme", "")),
        }

        # Judge scores
        for category, result in eval_results.items():
            result_row[f"{category}_score"] = result.score

        # Reference scores from dataset
        for col in REFERENCE_COLS:
            result_row[f"ref_{col}"] = row.get(col, None)

        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    csv_content = results_df.to_csv(index=False)

    return {
        "success": True,
        "output_file": output_path,
        "entries_evaluated": len(results),
        "total_entries": total_entries,
        "csv_content": csv_content,
        "message": f"Story writing benchmark evaluation completed. Results saved to {output_path}",
    }


if __name__ == "__main__":
    mcp.run()
