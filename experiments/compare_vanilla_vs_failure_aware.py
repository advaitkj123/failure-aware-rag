# experiments/compare_vanilla_vs_failure_aware.py

import json
import pandas as pd
from pathlib import Path

from features.answer_instability import logical_instability

INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/table_system_comparison.csv"


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for r in records:

        baseline_answer = r["baseline_answer"]
        vanilla_rag_answer = r["vanilla_rag_answer"]
        failure_aware_answer = r["failure_aware_answer"]

        # Vanilla RAG vs Baseline
        vanilla_logic = logical_instability(
            baseline_answer,
            vanilla_rag_answer
        )

        # Failure-Aware vs Baseline
        failure_logic = logical_instability(
            baseline_answer,
            failure_aware_answer
        )

        rows.append({
            "entailment_vanilla": vanilla_logic["entailment"],
            "contradiction_vanilla": vanilla_logic["contradiction"],

            "entailment_failure": failure_logic["entailment"],
            "contradiction_failure": failure_logic["contradiction"],

            "used_retrieval_failure_aware": r["used_retrieval"]
        })

    df = pd.DataFrame(rows)

    summary = {
        "Total Queries": len(df),
        "Vanilla RAG: Retrieval Used (%)": 100.0,
        "Failure-Aware RAG: Retrieval Used (%)":
            100.0 * df["used_retrieval_failure_aware"].mean(),

        "Mean Entailment (Vanilla RAG)":
            df["entailment_vanilla"].mean(),

        "Mean Entailment (Failure-Aware)":
            df["entailment_failure"].mean(),

        "Mean Contradiction (Vanilla RAG)":
            df["contradiction_vanilla"].mean(),

        "Mean Contradiction (Failure-Aware)":
            df["contradiction_failure"].mean(),
    }

    summary_df = pd.DataFrame([summary])

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print(summary_df)
    print(f"[OK] Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
