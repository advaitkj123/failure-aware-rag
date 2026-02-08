import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # generation uses GPU only

import json
from pathlib import Path

from generation.generate import generate_answer
from retrieval.bm25_retriever import build_retriever
from features.answer_instability import semantic_instability
from policy.gate import should_retrieve

OUTPUT_PATH = Path("results/vanilla_vs_failure_aware.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    ("q10", "Who was the first President of the United States?")
]

def main():
    retriever = build_retriever(limit=200)
    results = []

    for qid, query in QUERIES:
        # Baseline generation (GPU)
        ans_no = generate_answer(query)

        # Retrieve + RAG generation (GPU, limited context)
        passages = retriever.retrieve(query, k=2)  # reduced from 3
        ans_rag = generate_answer(query, passages)

        # Instability (CPU-only, cheap)
        instability = semantic_instability(ans_no, ans_rag)

        # Failure-aware decision (no extra generation)
        use_retrieval = should_retrieve(instability)
        ans_failure_aware = ans_rag if use_retrieval else ans_no

        results.append({
            "qid": qid,
            "query": query,
            "semantic_instability": instability,
            "vanilla_rag_answer": ans_rag,
            "failure_aware_answer": ans_failure_aware,
            "retrieval_used": use_retrieval
        })

        print(f"[OK] {qid} | retrieve={use_retrieval}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
