import json
import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_JSON = "results/ablation_tau.json"
OUTPUT_CSV = "results/table_ablation_tau.csv"


def main():
    input_path = Path(INPUT_JSON)
    assert input_path.exists(), "Run experiments.ablation_threshold first"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Clean, IEEE-friendly column names
    df.rename(columns={
        "percentile": "Gate Percentile",
        "gate_threshold": "Gate Threshold (τ)",
        "queries_where_retrieval_skipped": "Queries Where Retrieval Skipped",
        "total_queries": "Total Queries"
    }, inplace=True)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("[OK] Ablation table written to", OUTPUT_CSV)
    print(df)


if __name__ == "__main__":
    main()
