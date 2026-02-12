import json
import numpy as np
from pathlib import Path

INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_PATH = Path("results/table_ablation_tau.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PERCENTILES = [60, 70, 75, 80, 85, 90]


def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    instabilities = np.array(
        [r["semantic_instability"] for r in records]
    )

    rows = []

    for p in PERCENTILES:

        tau = float(np.percentile(instabilities, p))

        blocked = sum(
            1 for r in records
            if r["semantic_instability"] <= tau
        )

        rows.append({
            "Gate Percentile": p,
            "Threshold": tau,
            "Retrieval Skipped": blocked,
            "Total Queries": len(records)
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(df)
    print(f"[OK] Ablation table saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
