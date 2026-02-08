# experiments/ablation_threshold.py
import json
from pathlib import Path

from retrieval.bm25_retriever import build_retriever
from generation.generate import generate_answer
from features.answer_instability import semantic_instability
from policy.gate import should_retrieve

THRESHOLDS = [0.10, 0.15, 0.18, 0.22, 0.30]

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

    # --- Stage 1: GPU generation ONCE ---
    cache = []

    for qid, query in QUERIES:
        ans_no = generate_answer(query)

        passages = retriever.retrieve(query, k=2)  # GPU-safe context
        ans_rag = generate_answer(query, passages)

        instability = semantic_instability(ans_no, ans_rag)

        cache.append({
            "qid": qid,
            "query": query,
            "instability": instability
        })

    # --- Stage 2: CPU-only threshold sweep ---
    summary = []

    for tau in THRESHOLDS:
        blocked = sum(
            1 for r in cache
            if not should_retrieve(r["instability"], threshold=tau)
        )

        summary.append({
            "threshold": tau,
            "blocked_queries": blocked,
            "total_queries": len(QUERIES)
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)

if __name__ == "__main__":
    main()
