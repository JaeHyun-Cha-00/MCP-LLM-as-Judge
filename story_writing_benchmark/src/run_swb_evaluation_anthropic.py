import argparse
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from clients_anthropic import AnthropicClient
from evaluation import StoryEvaluator

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"

REFERENCE_COLS = [f"q{i}" for i in range(1, 16)]


def main():
    load_dotenv(SCRIPT_DIR.parent.parent / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N entries")
    args = parser.parse_args()

    client = AnthropicClient()
    evaluator = StoryEvaluator(client)

    dataset = pd.read_csv(str(DATASET_PATH))
    if args.limit:
        dataset = dataset.head(args.limit)
    total = len(dataset)
    print(f"[INFO] Loaded {total} entries from {DATASET_PATH}")

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(RESULTS_DIR / f"swb_claude_sonnet_4.6_{timestamp}.csv")

    results = []
    for i in range(total):
        print(f"[INFO] Evaluating entry {i + 1}/{total}")
        row = dataset.iloc[i]
        story = str(row.get("story_text", ""))
        model = str(row.get("model_name", ""))

        eval_results = evaluator.evaluate_all_categories(story)
        time.sleep(1)

        result_row = {
            "index": i,
            "prompt_id": str(row.get("prompt_id", "")),
            "model": model,
            "language": str(row.get("language", "")),
            "theme": str(row.get("theme", "")),
        }
        for category, result in eval_results.items():
            result_row[f"{category}_score"] = result.score
        for col in REFERENCE_COLS:
            result_row[f"ref_{col}"] = row.get(col, None)

        results.append(result_row)

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"[INFO] Checkpoint saved at entry {i + 1}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
