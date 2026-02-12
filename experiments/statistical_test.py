# experiments/statistical_test.py

import pandas as pd
from scipy.stats import ttest_rel

INPUT_CSV = "results/answer_instability_labeled.csv"

def main():

    df = pd.read_csv(INPUT_CSV)

    # Your current columns are:
    # semantic_instability, entailment, contradiction

    # We interpret:
    # entailment = baseline → vanilla
    # harm proxy = contradiction

    vanilla_entailment = df["entailment"]
    vanilla_contradiction = df["contradiction"]

    print("=== Paired t-test (Entailment vs 0.5 baseline) ===")
    t_stat, p_val = ttest_rel(vanilla_entailment, [0.5]*len(df))
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6f}")

    print("\n=== Mean Metrics ===")
    print(f"Mean Entailment: {vanilla_entailment.mean():.4f}")
    print(f"Mean Contradiction: {vanilla_contradiction.mean():.4f}")

if __name__ == "__main__":
    main()
