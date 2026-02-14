# experiments/statistical_test.py

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel

INPUT_PATH = "results/failure_aware_outputs.json"


def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1)


def main():

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # --------------------------------------------
    # Extract paired metrics
    # --------------------------------------------
    vanilla_ent = []
    failure_ent = []

    vanilla_contra = []
    failure_contra = []

    for r in data:
        vanilla_ent.append(r["vanilla_entailment"])
        failure_ent.append(r["failure_entailment"])

        vanilla_contra.append(r["vanilla_contradiction"])
        failure_contra.append(r["failure_contradiction"])

    vanilla_ent = np.array(vanilla_ent)
    failure_ent = np.array(failure_ent)

    vanilla_contra = np.array(vanilla_contra)
    failure_contra = np.array(failure_contra)

    # --------------------------------------------
    # Paired t-tests
    # --------------------------------------------
    t_ent, p_ent = ttest_rel(failure_ent, vanilla_ent)
    t_contra, p_contra = ttest_rel(failure_contra, vanilla_contra)

    # --------------------------------------------
    # Effect size
    # --------------------------------------------
    d_ent = cohens_d(failure_ent, vanilla_ent)
    d_contra = cohens_d(vanilla_contra, failure_contra)

    print("\n=== IEEE Statistical Evaluation ===\n")

    print("Mean Entailment:")
    print("Vanilla:", vanilla_ent.mean())
    print("Failure-Aware:", failure_ent.mean())
    print("Paired t-test p-value:", p_ent)
    print("Cohen's d:", d_ent)
    print()

    print("Mean Contradiction:")
    print("Vanilla:", vanilla_contra.mean())
    print("Failure-Aware:", failure_contra.mean())
    print("Paired t-test p-value:", p_contra)
    print("Cohen's d:", d_contra)
    print()

    print("Interpretation:")
    if p_ent < 0.05:
        print("✓ Entailment improvement statistically significant")
    else:
        print("✗ Entailment NOT statistically significant")

    if p_contra < 0.05:
        print("✓ Contradiction reduction statistically significant")
    else:
        print("✗ Contradiction reduction NOT statistically significant")


if __name__ == "__main__":
    main()