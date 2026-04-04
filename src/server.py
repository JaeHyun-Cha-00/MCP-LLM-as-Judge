from fastmcp import FastMCP
from evaluation import StoryEvaluator, STORY_EVALUATION_CATEGORIES
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
def read_dataset(index: int) -> dict:
    """Read a single entry from the dataset by index (0-based). Returns the model name and response text."""
    print(f"[INFO] Tool called: read_dataset with index={index}")
    dataset = load_dataset()
    
    if dataset.empty:
        return {"error": "Dataset is not loaded or is empty"}
    
    if index < 0 or index >= len(dataset):
        return {
            "error": f"Index {index} out of range. Dataset has {len(dataset)} entries (valid range: 0-{len(dataset)-1})."
        }
    
    row = dataset.iloc[index]
    return {
        "index": index,
        "model": str(row.get("model", "")),
        "response": str(row.get("response", "")),
        "total_entries": len(dataset)
    }

@mcp.tool()
def list_categories() -> list[str]:
    """Return all supported evaluation categories."""
    print("[INFO] Tool called: list_categories")
    return STORY_EVALUATION_CATEGORIES

@mcp.tool()
def evaluate_all_categories(story: str) -> dict[str, dict]:
    """Evaluate a story across all evaluation categories."""
    print("[INFO] Tool called: evaluate_all_categories")
    results = evaluator.evaluate_all_categories(story)
    return {cat: res.to_dict() for cat, res in results.items()}

@mcp.tool()
def evaluate_creativity(story: str) -> dict:
    """Evaluate a story's creativity directly without breaking it into categories."""
    print("[INFO] Tool called: evaluate_creativity")
    result = evaluator.evaluate_creativity(story)
    return result.to_dict()

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
        
        # 1. Standalone creativity evaluation (direct, without category context)
        standalone_creativity = evaluator.evaluate_creativity(story)
        
        # 2. Evaluate all categories (includes contextual creativity)
        eval_results = evaluator.evaluate_all_categories(story)
        contextual_creativity = eval_results.get("Creativity", standalone_creativity)
        
        # 3. Analyze which categories influenced the creativity difference (like compare_creativity_scores)
        analysis = evaluator.analyze_creativity_difference(
            story,
            standalone_creativity,
            eval_results
        )
        
        # Create result row
        result_row = {
            "index": i,
            "model": model,
        }
        
        # Add all category scores (only scores, no explanations)
        for category, result in eval_results.items():
            if category != "Creativity":  # Creativity is handled separately
                result_row[f"{category}_score"] = result.score
        
        # Add both creativity scores
        result_row["creativity_standalone_score"] = standalone_creativity.score
        result_row["creativity_contextual_score"] = contextual_creativity.score
        result_row["creativity_difference"] = round(abs(standalone_creativity.score - contextual_creativity.score), 1)
        
        # Add analysis results (influential categories)
        influential_categories = analysis.get("influential_categories", [])
        result_row["influential_categories"] = ", ".join(influential_categories) if influential_categories else ""
        
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