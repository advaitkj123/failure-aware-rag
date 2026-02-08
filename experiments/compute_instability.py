import json
import csv
from pathlib import Path
import os

# Force CPU (GPU-safe)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from features.answer_instability import (
    semantic_instability,
    logical_instability,
    structural_drift,
)

# -----------------------------
# Config
# -----------------------------
INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/answer_instability.csv"


def main():
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for r in records:
        qid = r.get("qid", "")
        query = r["query"]

        # --- Canonical answers ---
        baseline_answer = r["baseline_answer"]
        vanilla_rag_answer = r["vanilla_rag_answer"]
        failure_aware_answer = r["failure_aware_answer"]

        # --- Instability: Vanilla RAG vs Baseline ---
        sem_instab = semantic_instability(
            baseline_answer,
            vanilla_rag_answer
        )
        logic = logical_instability(
            baseline_answer,
            vanilla_rag_answer
        )
        struct = structural_drift(
            baseline_answer,
            vanilla_rag_answer
        )

        row = {
            "qid": qid,
            "query": query,
            "semantic_instability": sem_instab,
            "entailment": logic["entailment"],
            "neutral": logic["neutral"],
            "contradiction": logic["contradiction"],
            "length_diff": struct["length_diff"],
            "length_ratio": struct["length_ratio"],
            "used_retrieval": r["used_retrieval"],
        }

        rows.append(row)

    # -----------------------------
    # Write CSV
    # -----------------------------
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
