import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "results/answer_instability_labeled.csv"
OUTPUT_PNG = "results/instability_vs_harm.png"


def main():
    df = pd.read_csv(INPUT_CSV)

    # Split by system behavior
    vanilla_used = df[df["used_retrieval"] == True]
    retrieval_skipped = df[df["used_retrieval"] == False]

    plt.figure(figsize=(6.5, 4.5))

    # Vanilla RAG points
    plt.scatter(
        vanilla_used["semantic_instability"],
        vanilla_used["entailment"],
        c="tab:red",
        label="Vanilla RAG (retrieval used)",
        alpha=0.85
    )

    # Failure-Aware RAG points
    plt.scatter(
        retrieval_skipped["semantic_instability"],
        retrieval_skipped["entailment"],
        c="tab:blue",
        label="Failure-Aware RAG (retrieval skipped)",
        alpha=0.85
    )

    plt.xlabel("Semantic Instability (Baseline vs Vanilla RAG)")
    plt.ylabel("Entailment (Baseline → Vanilla RAG)")
    plt.title("Failure-Aware RAG vs Vanilla RAG")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()

    print(f"[OK] Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
