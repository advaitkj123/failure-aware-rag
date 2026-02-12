import json
import numpy as np
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
# -----------------------------
# Dynamic Query Builder (IEEE-scale)
# -----------------------------
from retrieval.bm25_retriever import load_wiki_corpus

def build_query_set(n=200):
    corpus = load_wiki_corpus(limit=n * 2)

    queries = []
    for i, text in enumerate(corpus[:n]):
        clean = text.replace("\n", " ").strip()
        short = clean[:120]
        queries.append((f"q{i+1}", short + "?"))

    return queries

QUERIES = build_query_set(200)



# -----------------------------
# Main experiment
# -----------------------------
def main():
    retriever = build_retriever(limit=200)
    raw_results = []

    # ---------- PASS 1: generate + measure instability ----------
    for qid, query in QUERIES:
        # Baseline (NO retrieval)
        baseline_answer = generate_answer(query)

        # Vanilla RAG (ALWAYS retrieve)
        passages = retriever.retrieve(query, k=2)
        vanilla_rag_answer = generate_answer(query, passages)

        # Instability (CPU)
        instability = compute_semantic_instability(
            baseline_answer,
            vanilla_rag_answer
        )

        raw_results.append({
            "qid": qid,
            "query": query,
            "baseline_answer": baseline_answer,
            "vanilla_rag_answer": vanilla_rag_answer,
            "semantic_instability": instability
        })

    # ---------- Dynamic gate threshold (relative, IEEE-safe) ----------
    instability_values = [r["semantic_instability"] for r in raw_results]
    gate_threshold = float(np.quantile(instability_values, 0.80))

    print(f"[INFO] Dynamic gate threshold (80th percentile): {gate_threshold:.6f}")

    # ---------- PASS 2: failure-aware selection ----------
    final_results = []

    for r in raw_results:
        use_retrieval = should_retrieve(
            r["semantic_instability"],
            gate_threshold
        )

        failure_aware_answer = (
            r["vanilla_rag_answer"]
            if use_retrieval
            else r["baseline_answer"]
        )

        explanation = explain_decision(
            r["semantic_instability"],
            gate_threshold
        )

        final_results.append({
            "qid": r["qid"],
            "query": r["query"],
            "semantic_instability": r["semantic_instability"],

            # --- pipeline outputs ---
            "baseline_answer": r["baseline_answer"],
            "vanilla_rag_answer": r["vanilla_rag_answer"],
            "failure_aware_answer": failure_aware_answer,

            # --- gate ---
            "used_retrieval": use_retrieval,
            "decision_explanation": explanation
        })

        print(f"[OK] {r['qid']} | retrieve={use_retrieval}")

    # -----------------------------
    # Save results
    # -----------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"[DONE] Saved failure-aware outputs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
