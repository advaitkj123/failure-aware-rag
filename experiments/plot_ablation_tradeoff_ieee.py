# experiments/plot_ablation_tradeoff_ieee.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "results/table_ablation_tau.csv"
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ieee_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "figure.dpi": 300
    })

def main():
    ieee_style()

    df = pd.read_csv(INPUT_CSV)

    retrieval_usage = 100 - (df["Retrieval Skipped"] / df["Total Queries"] * 100)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(df["Gate Percentile"], retrieval_usage, marker="o")

    ax.set_xlabel("Gate Percentile")
    ax.set_ylabel("Retrieval Usage (%)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "retrieval_tradeoff_ieee.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "retrieval_tradeoff_ieee.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
