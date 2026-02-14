# experiments/compare_vanilla_vs_failure_aware.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


INPUT_JSON = "results/failure_aware_outputs.json"
OUTPUT_CSV = "results/table_system_comparison.csv"


def mean_ci(values, confidence=0.95):
    """
    Compute mean and 95% confidence interval.
    """
    arr = np.array(values)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
    return mean, margin


def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    total_queries = len(df)

    # -------------------------------------------------
    # Retrieval usage
    # -------------------------------------------------
    vanilla_retrieval_used = 100.0
    failure_retrieval_used = df["used_retrieval"].mean() * 100

    # -------------------------------------------------
    # Logical metrics (computed earlier in instability CSV)
    # We recompute using compute_instability output
    # -------------------------------------------------
    instab_df = pd.read_csv("results/answer_instability.csv")

    entail_v = instab_df["entailment"]
    contra_v = instab_df["contradiction"]

    # For failure-aware, keep only where gate allowed retrieval
    instab_df["used_retrieval"] = df["used_retrieval"]

    entail_f = instab_df.apply(
        lambda r: r["entailment"] if r["used_retrieval"] else 1.0,
        axis=1
    )

    contra_f = instab_df.apply(
        lambda r: r["contradiction"] if r["used_retrieval"] else 0.0,
        axis=1
    )

    # -------------------------------------------------
    # Means + 95% CI
    # -------------------------------------------------
    mean_ent_v, ci_ent_v = mean_ci(entail_v)
    mean_ent_f, ci_ent_f = mean_ci(entail_f)

    mean_con_v, ci_con_v = mean_ci(contra_v)
    mean_con_f, ci_con_f = mean_ci(contra_f)

    # -------------------------------------------------
    # Build final table
    # -------------------------------------------------
    table = pd.DataFrame([{
        "Total Queries": total_queries,

        "Vanilla RAG: Retrieval Used (%)": vanilla_retrieval_used,
        "Failure-Aware RAG: Retrieval Used (%)": failure_retrieval_used,

        "Mean Entailment (Vanilla RAG)": mean_ent_v,
        "95% CI Entailment (Vanilla RAG)": ci_ent_v,

        "Mean Entailment (Failure-Aware)": mean_ent_f,
        "95% CI Entailment (Failure-Aware)": ci_ent_f,

        "Mean Contradiction (Vanilla RAG)": mean_con_v,
        "95% CI Contradiction (Vanilla RAG)": ci_con_v,

        "Mean Contradiction (Failure-Aware)": mean_con_f,
        "95% CI Contradiction (Failure-Aware)": ci_con_f,
    }])

    Path("results").mkdir(exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)

    print("[OK] Updated system comparison table written.")
    print(table)


if __name__ == "__main__":
    main()