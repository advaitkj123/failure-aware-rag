import pandas as pd
from pathlib import Path

INPUT_CSV = "results/answer_instability.csv"
OUTPUT_CSV = "results/answer_instability_labeled.csv"

def label_harm(row):
    """
    Simple, interpretable harm proxy.
    Harm = 1 if retrieval degrades logical confidence.
    """
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
        "harm"
    ]])

if __name__ == "__main__":
    main()
