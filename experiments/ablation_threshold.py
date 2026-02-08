import json
import numpy as np
from pathlib import Path

from retrieval.bm25_retriever import build_retriever
from generation.generate import generate_answer
from features.answer_instability import semantic_instability
from policy.gate import should_retrieve

# -----------------------------
# Percentile sweep (IEEE-clean)
# -----------------------------
PERCENTILES = [0.6, 0.7, 0.8, 0.9]

QUERIES = [
    ("q1", "Who discovered penicillin?"),
    ("q2", "What is the capital of Australia?"),
    ("q3", "Who wrote Hamlet?"),
    ("q4", "What year did the Titanic sink?"),
    ("q5", "Is Pluto a planet?")
]

OUTPUT_PATH = Path("results/ablation_tau.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    retriever = build_retriever(limit=200)

    # -----------------------------
    # Stage 1: GPU generation ONCE
    # -----------------------------
    cache = []

    for qid, query in QUERIES:
        baseline_answer = generate_answer(query)

        passages = retriever.retrieve(query, k=2)
        vanilla_rag_answer = generate_answer(query, passages)

        instability = semantic_instability(
            baseline_answer,
            vanilla_rag_answer
        )

        cache.append(instability)

    instabilities = np.array(cache)

    # -----------------------------
    # Stage 2: CPU-only ablation
    # -----------------------------
    summary = []

    for p in PERCENTILES:
        tau = float(np.quantile(instabilities, p))

        blocked = sum(
            1 for instab in instabilities
            if not should_retrieve(instab, tau)
        )

        summary.append({
            "percentile": p,
            "gate_threshold": tau,
            "queries_where_retrieval_skipped": blocked,
            "total_queries": len(QUERIES)
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)


if __name__ == "__main__":
    main()
