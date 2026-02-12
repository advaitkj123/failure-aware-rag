# experiments/statistical_test.py

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

INPUT_CSV = "results/answer_instability_labeled.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    vanilla = df["entailment_vanilla"]
    failure = df["entailment_failure_aware"]

    t_stat, t_p = ttest_rel(vanilla, failure)
    w_stat, w_p = wilcoxon(vanilla, failure)

    print("Paired t-test p-value:", t_p)
    print("Wilcoxon p-value:", w_p)

if __name__ == "__main__":
    main()
