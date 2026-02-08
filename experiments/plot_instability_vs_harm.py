import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "results/answer_instability_labeled.csv"
OUTPUT_PNG = "results/instability_vs_harm.png"

def main():
    df = pd.read_csv(INPUT_CSV)

    plt.figure(figsize=(6, 4))

    for label, grp in df.groupby("harm"):
        name = "Harmful" if label == 1 else "Benign"
        plt.scatter(
            grp["semantic_instability"],
            grp["entailment"],
            label=name,
            alpha=0.85
        )

    plt.xlabel("Semantic Instability")
    plt.ylabel("Entailment (Failure-Aware vs Vanilla RAG)")
    plt.title("Retrieval Harm vs Answer Instability")
    plt.legend()
    plt.tight_layout()

    Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print(f"[OK] Saved plot to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
