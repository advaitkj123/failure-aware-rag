# experiments/build_query_set.py

from datasets import load_dataset
import json
from pathlib import Path

OUTPUT_PATH = Path("data/processed/query_set_200.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    dataset = load_dataset("natural_questions", split="train", streaming=True)

    queries = []
    count = 0

    for ex in dataset:
        if "question" in ex and ex["question"] and len(ex["question"]) < 200:
            queries.append({
                "qid": f"q{count+1}",
                "query": ex["question"]
            })
            count += 1

        if count >= 200:
            break

    with open(OUTPUT_PATH, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"[OK] Saved {len(queries)} queries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
