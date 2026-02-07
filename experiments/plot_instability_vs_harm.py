import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/answer_instability_labeled.csv")

plt.figure(figsize=(6, 4))

for label, grp in df.groupby("harm"):
    name = "Harmful" if label == 1 else "Benign"
    plt.scatter(
        grp["semantic_instability"],
        grp["entailment"],
        label=name,
        alpha=0.8
    )

plt.xlabel("Semantic Instability")
plt.ylabel("Entailment (RAG vs No-RAG)")
plt.title("Retrieval Harm vs Answer Instability")
plt.legend()
plt.tight_layout()
plt.show()
