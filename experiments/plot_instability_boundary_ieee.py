# experiments/plot_instability_boundary_ieee.py

import json
import pandas as pd
import matplotlib.pyplot as plt
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

    threshold = df["semantic_instability"].quantile(0.75)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    retrieved = df[df["used_retrieval"] == True]
    skipped = df[df["used_retrieval"] == False]

    ax.scatter(skipped["semantic_instability"], skipped["failure_entailment"],
               label="Retrieval Skipped", alpha=0.7)

    ax.scatter(retrieved["semantic_instability"], retrieved["failure_entailment"],
               label="Retrieval Used", alpha=0.7)

    ax.axvline(threshold, linestyle="--", label="Gate Threshold")

    ax.set_xlabel("Semantic Instability")
    ax.set_ylabel("Failure-Aware Entailment")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "instability_boundary_ieee.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "instability_boundary_ieee.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
