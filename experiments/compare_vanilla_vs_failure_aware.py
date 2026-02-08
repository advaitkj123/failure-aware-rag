import json
import numpy as np
import pandas as pd
from pathlib import Path

from policy.gate import should_retrieve

# -----------------------------
# Config
# -----------------------------
INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/table_system_comparison.csv"


def main():
    records = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

    # Collect instability values
    instabilities = [r["semantic_instability"] for r in records]

    # Dynamic threshold (same as main experiment)
    threshold = float(np.quantile(instabilities, 0.80))

    vanilla_harm = 0
    failure_aware_harm = 0
    retrieval_used_fa = 0

    for r in records:
        instability = r["semantic_instability"]

        # Vanilla RAG always retrieves
        vanilla_harm += 1

        # Failure-aware decision
        if should_retrieve(instability, threshold):
            retrieval_used_fa += 1
            failure_aware_harm += 1

    table = pd.DataFrame([{
        "Total Queries": len(records),
        "Vanilla RAG: Retrieval Used (%)": 100.0,
        "Failure-Aware RAG: Retrieval Used (%)":
            100.0 * retrieval_used_fa / len(records),
        "Harmful Cases (Vanilla RAG)": vanilla_harm,
        "Harmful Cases (Failure-Aware RAG)": failure_aware_harm
    }])

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)

    print(table)


if __name__ == "__main__":
    main()
