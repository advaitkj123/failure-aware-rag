import json
import csv
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from features.answer_instability import (
    semantic_instability,
    logical_instability,
    structural_drift,
)

# ---- CONFIG ----
from pathlib import Path
import json

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
        ans_no = r["vanilla_answer"]
        ans_rag = r["rag_answer"]


        # --- Instability metrics ---
        sem_instab = semantic_instability(ans_no, ans_rag)
        logic = logical_instability(ans_no, ans_rag)
        struct = structural_drift(ans_no, ans_rag)

        row = {
            "qid": qid,
            "query": query,
            "semantic_instability": sem_instab,
            "entailment": logic["entailment"],
            "neutral": logic["neutral"],
            "contradiction": logic["contradiction"],
            "length_diff": struct["length_diff"],
            "length_ratio": struct["length_ratio"],
        }

        rows.append(row)

    # ---- Write CSV ----
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    main()
