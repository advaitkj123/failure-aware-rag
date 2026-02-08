
import json
from pathlib import Path

from retrieval.bm25_retriever import build_retriever
from generation.generate import generate_answer
from features.answer_instability import semantic_instability
from policy.gate import should_retrieve


OUTPUT_PATH = Path("results/failure_aware_outputs.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)


def main():
    retriever = build_retriever(limit=500)

    queries = [
        "Who wrote the novel 1984?",
        "What is the capital of France?",
        "Who developed the theory of relativity?"
    ]

    results = []

    for qid, query in enumerate(queries):
        ans_vanilla = generate_answer(query)
        instability_probe = semantic_instability(ans_vanilla, ans_vanilla)

        use_retrieval = should_retrieve(instability_probe)

        passages = retriever.retrieve(query) if use_retrieval else None
        ans_rag = generate_answer(query, passages)

        instability = semantic_instability(ans_vanilla, ans_rag)

        results.append({
            "id": qid,
            "query": query,
            "vanilla_answer": ans_vanilla,
            "rag_answer": ans_rag,
            "semantic_instability": instability,
            "retrieval_used": use_retrieval
        })

        print(f"[OK] q{qid} | retrieve={use_retrieval}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
