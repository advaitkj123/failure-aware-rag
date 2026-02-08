import json
import pandas as pd
from pathlib import Path

INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/table_example_comparisons.csv"


def main():
    records = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

    rows = []
    for r in records:
        if not r["used_retrieval"]:
            rows.append({
                "qid": r["qid"],
                "query": r["query"],
                "vanilla_rag_answer": r["vanilla_rag_answer"][:200],
                "failure_aware_answer": r["failure_aware_answer"][:200],
                "semantic_instability": r["semantic_instability"],
                "decision_explanation": r["decision_explanation"],
            })

    df = pd.DataFrame(rows)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Wrote {len(df)} examples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
