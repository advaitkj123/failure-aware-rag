# experiments/plot_effect_size_distribution_ieee.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_JSON = "results/failure_aware_outputs.json"
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

    with open(INPUT_JSON) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Per-query entailment gain
    df["entailment_gain"] = df["failure_entailment"] - df["vanilla_entailment"]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.hist(df["entailment_gain"], bins=25)

    ax.axvline(df["entailment_gain"].mean(), linestyle="--",
               label="Mean Gain")

    ax.set_xlabel("Per-Query Entailment Gain")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "effect_size_distribution_ieee.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "effect_size_distribution_ieee.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
