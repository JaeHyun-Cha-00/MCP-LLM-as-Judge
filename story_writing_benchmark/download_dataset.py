"""Download lars1234/story_writing_benchmark from HuggingFace and save to dataset/data.csv."""
from datasets import load_dataset
from pathlib import Path
import pandas as pd

DATASET_PATH = Path(__file__).parent / "dataset" / "data.csv"

def main():
    print("Downloading lars1234/story_writing_benchmark ...")
    ds = load_dataset("lars1234/story_writing_benchmark")
    split = ds["train"]
    df = pd.DataFrame(split)
    df = df[df["language"] == "en"].reset_index(drop=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} English rows to {DATASET_PATH}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
