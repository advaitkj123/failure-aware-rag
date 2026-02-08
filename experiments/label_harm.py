import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "results/answer_instability.csv"
OUTPUT_CSV = "results/answer_instability_labeled.csv"


def label_harm(row):
    """
    Harm is defined ONLY when retrieval is used
    and logical degradation is observed.
    """
    if not row["used_retrieval"]:
        return 0

    if row["contradiction"] > 0.1:
        return 1
    if row["entailment"] < 0.6:
        return 1
    if row["neutral"] > 0.5:
        return 1
    return 0


def main():
    df = pd.read_csv(INPUT_CSV)

    df["harm"] = df.apply(label_harm, axis=1)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("[OK] Harm labels added")
    print(df[[
        "qid",
        "semantic_instability",
        "entailment",
        "neutral",
        "contradiction",
        "used_retrieval",
        "harm"
    ]])

if __name__ == "__main__":
    main()
