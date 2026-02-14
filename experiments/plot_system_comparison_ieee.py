# experiments/plot_system_comparison_ieee.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_CSV = "results/table_system_comparison.csv"
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ieee_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "figure.dpi": 300
    })

def plot_metric(metric_v, metric_f, ci_v, ci_f, ylabel, filename):
    systems = ["Vanilla RAG", "Failure-Aware RAG"]
    means = [metric_v, metric_f]
    cis = [ci_v, ci_f]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(systems, means, yerr=cis, capsize=5)

    ax.set_ylabel(ylabel)
    ax.set_title("")
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()

def main():
    ieee_style()

    df = pd.read_csv(INPUT_CSV)

    plot_metric(
        df["Mean Entailment (Vanilla RAG)"][0],
        df["Mean Entailment (Failure-Aware)"][0],
        df["95% CI Entailment (Vanilla RAG)"][0],
        df["95% CI Entailment (Failure-Aware)"][0],
        "Mean Entailment",
        "entailment_comparison_ieee"
    )

    plot_metric(
        df["Mean Contradiction (Vanilla RAG)"][0],
        df["Mean Contradiction (Failure-Aware)"][0],
        df["95% CI Contradiction (Vanilla RAG)"][0],
        df["95% CI Contradiction (Failure-Aware)"][0],
        "Mean Contradiction",
        "contradiction_comparison_ieee"
    )

if __name__ == "__main__":
    main()
