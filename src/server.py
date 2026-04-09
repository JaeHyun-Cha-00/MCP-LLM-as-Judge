from fastmcp import FastMCP
from evaluation import StoryEvaluator
from clients import WolverineClient
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

mcp = FastMCP(name="MCP-LLM Judge")
client = WolverineClient()
evaluator = StoryEvaluator(client)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
# Project root is one level up from src/
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"
_dataset = None

def load_dataset():
    """Load the dataset into memory on first access."""
    global _dataset
    if _dataset is None:
        dataset_path_str = str(DATASET_PATH)
        if os.path.exists(dataset_path_str):
            try:
                _dataset = pd.read_csv(dataset_path_str)
                print(f"[INFO] Dataset loaded: {len(_dataset)} entries from {dataset_path_str}")
            except Exception as e:
                print(f"[ERROR] Failed to load dataset: {e}")
                _dataset = pd.DataFrame()
        else:
            print(f"[WARNING] Dataset file not found: {dataset_path_str}")
            print(f"[INFO] Current working directory: {os.getcwd()}")
            print(f"[INFO] Script directory: {SCRIPT_DIR}")
            _dataset = pd.DataFrame()
    return _dataset

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    try:
        os.makedirs(str(RESULTS_DIR), exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create results directory: {e}")

# Initialize results directory
ensure_results_dir()

@mcp.tool()
def evaluate_single_story(story: str) -> dict[str, dict]:
    """Evaluate a single story across all evaluation categories (for testing purposes)."""
    print("[INFO] Tool called: evaluate_single_story")
    results = evaluator.evaluate_all_categories(story)
    return {cat: res.to_dict() for cat, res in results.items()}

@mcp.tool()
def evaluate_full_dataset(output_filename: str = None) -> dict:
    """Evaluate the entire dataset and save results to CSV. Returns the CSV file path and summary."""
    print(f"[INFO] Tool called: evaluate_full_dataset")
    dataset = load_dataset()
    
    if dataset.empty:
        return {"error": "Dataset is not loaded or is empty"}
    
    total_entries = len(dataset)
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_results_full_{total_entries}_{timestamp}.csv"
    
    # Ensure results directory exists
    ensure_results_dir()
    output_path = str(RESULTS_DIR / output_filename)
    
    # Evaluate each entry
    results = []
    for i in range(total_entries):
        print(f"[INFO] Evaluating entry {i+1}/{total_entries}")
        row = dataset.iloc[i]
        story = str(row.get("response", ""))
        model = str(row.get("model", ""))

        eval_results = evaluator.evaluate_all_categories(story)

        result_row = {
            "index": i,
            "model": model,
        }
        for category, result in eval_results.items():
            result_row[f"{category}_score"] = result.score

        results.append(result_row)
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # Get CSV content as string
    csv_content = results_df.to_csv(index=False)
    
    return {
        "success": True,
        "output_file": output_path,
        "entries_evaluated": len(results),
        "total_entries": total_entries,
        "csv_content": csv_content,
        "message": f"Full dataset evaluation completed. Results saved to {output_path}"
    }

if __name__ == "__main__":
    mcp.run()