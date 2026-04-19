import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from clients import WolverineClient
from evaluation import StoryEvaluator

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"


def main():
    client = WolverineClient()
    evaluator = StoryEvaluator(client)

    dataset = pd.read_csv(str(DATASET_PATH))
    total = len(dataset)
    print(f"[INFO] Loaded {total} entries from {DATASET_PATH}")

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(RESULTS_DIR / f"evaluation_results_full_{total}_{timestamp}.csv")

    results = []
    for i in range(total):
        print(f"[INFO] Evaluating entry {i + 1}/{total}")
        row = dataset.iloc[i]
        story = str(row.get("response", ""))
        model = str(row.get("model", ""))

        eval_results = evaluator.evaluate_all_categories(story)

        result_row = {"index": i, "model": model}
        for category, result in eval_results.items():
            result_row[f"{category}_score"] = result.score

        results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
