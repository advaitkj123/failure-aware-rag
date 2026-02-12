# experiments/compare_vanilla_vs_failure_aware.py

import json
import pandas as pd
from pathlib import Path

from features.answer_instability import (
    semantic_instability,
    logical_instability,
    structural_drift,
)

INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/table_system_comparison.csv"


def compute_metrics(ans_a: str, ans_b: str):
    """
    Computes semantic + logical drift between two answers.
    """
    sem = semantic_instability(ans_a, ans_b)
    logic = logical_instability(ans_a, ans_b)

    return {
        "semantic_instability": sem,
        "entailment": logic["entailment"],
        "neutral": logic["neutral"],
        "contradiction": logic["contradiction"],
    }


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for r in records:
        qid = r["qid"]
        query = r["query"]

        vanilla_answer = r["vanilla_answer"]
        rag_answer = r["rag_answer"]
        failure_answer = r["failure_aware_answer"]

        # ---------------------------
        # Vanilla vs RAG
        # ---------------------------
        vanilla_vs_rag = compute_metrics(vanilla_answer, rag_answer)

        # ---------------------------
        # Vanilla vs Failure-Aware
        # ---------------------------
        vanilla_vs_failure = compute_metrics(vanilla_answer, failure_answer)

        rows.append({
            "qid": qid,
            "query": query,

            # --- Vanilla vs RAG ---
            "semantic_instability_vanilla_rag": vanilla_vs_rag["semantic_instability"],
            "entailment_vanilla_rag": vanilla_vs_rag["entailment"],
            "contradiction_vanilla_rag": vanilla_vs_rag["contradiction"],

            # --- Vanilla vs Failure-Aware ---
            "semantic_instability_vanilla_failure": vanilla_vs_failure["semantic_instability"],
            "entailment_vanilla_failure": vanilla_vs_failure["entailment"],
            "contradiction_vanilla_failure": vanilla_vs_failure["contradiction"],

            # --- Retrieval decision ---
            "used_retrieval_failure_aware": r["used_retrieval"],
        })

    df = pd.DataFrame(rows)

    # ---------------------------
    # IEEE-Style Summary Metrics
    # ---------------------------
    summary = {
        "Total Queries": len(df),
        "Vanilla RAG: Retrieval Used (%)": 100.0,
        "Failure-Aware RAG: Retrieval Used (%)":
            100.0 * df["used_retrieval_failure_aware"].mean(),
        "Mean Entailment (Vanilla RAG)":
            df["entailment_vanilla_rag"].mean(),
        "Mean Entailment (Failure-Aware)":
            df["entailment_vanilla_failure"].mean(),
        "Mean Contradiction (Vanilla RAG)":
            df["contradiction_vanilla_rag"].mean(),
        "Mean Contradiction (Failure-Aware)":
            df["contradiction_vanilla_failure"].mean(),
    }

    summary_df = pd.DataFrame([summary])

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print("[OK] System comparison table saved to", OUTPUT_CSV)
    print(summary_df)


if __name__ == "__main__":
    main()
