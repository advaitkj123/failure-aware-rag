# experiments/failure_aware_generation.py

import json
from pathlib import Path

from generation.generate import generate_answer
from retrieval.bm25_retriever import build_retriever
from features.answer_instability import semantic_instability as compute_semantic_instability
from policy.gate import should_retrieve
from policy.explain_gate import explain_decision


# -----------------------------
# Output setup
# -----------------------------
OUTPUT_PATH = Path("results/failure_aware_outputs.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Evaluation queries
# -----------------------------
QUERIES = [
    ("q1", "Who discovered penicillin?"),
    ("q2", "What is the capital of Australia?"),
    ("q3", "Who wrote Hamlet?"),
    ("q4", "What year did the Titanic sink?"),
    ("q5", "Is Pluto a planet?"),
    ("q6", "Who invented the telephone?"),
    ("q7", "What is the boiling point of water?"),
    ("q8", "Who painted the Mona Lisa?"),
    ("q9", "What is the largest planet in the Solar System?"),
    ("q10", "Who was the first President of the United States?"),
]


# -----------------------------
# Main experiment
# -----------------------------
def main():
    retriever = build_retriever(limit=200)
    results = []

    for qid, query in QUERIES:
        # Baseline generation (NO retrieval) — GPU
        answer_no = generate_answer(query)

        # Candidate RAG generation — GPU (bounded context)
        passages = retriever.retrieve(query, k=3)
        answer_rag = generate_answer(query, passages)

        # Semantic instability — CPU
        instability = compute_semantic_instability(answer_no, answer_rag)

        # Failure-aware gate — CPU
        use_retrieval = should_retrieve(instability)

        # Final answer selection (NO extra generation)
        final_answer = answer_rag if use_retrieval else answer_no

        #  Human-readable explanation — CPU
        explanation = explain_decision(instability)

        results.append({
            "qid": qid,
            "query": query,
            "semantic_instability": instability,
            "used_retrieval": use_retrieval,
            "decision_explanation": explanation,
            "final_answer": final_answer
        })

        print(f"[OK] {qid} | retrieve={use_retrieval}")

    # -----------------------------
    # Save results
    # -----------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved failure-aware outputs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
