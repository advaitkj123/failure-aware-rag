import json
import numpy as np
from pathlib import Path

from generation.generate import generate_answer
from retrieval.bm25_retriever import build_retriever
from features.answer_instability import semantic_instability
from policy.gate import should_retrieve
from policy.explain_gate import explain_decision


# -----------------------------
# Config
# -----------------------------
QUERY_PATH = "data/processed/query_set_200.json"
OUTPUT_PATH = Path("results/failure_aware_outputs.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GATE_PERCENTILE = 75
CALIBRATION_SPLIT = 0.7   # 70% calibration, 30% evaluation
TOP_K = 2                 # retrieval depth (GPU safe)


# -----------------------------
# Load queries
# -----------------------------
with open(QUERY_PATH, "r", encoding="utf-8") as f:
    QUERY_SET = json.load(f)


# -----------------------------
# Main
# -----------------------------
def main():

    retriever = build_retriever()

    raw_results = []

    print(f"[INFO] Loaded {len(QUERY_SET)} queries")

    # ==========================================================
    # PASS 1 — Generate baseline + vanilla RAG + instability
    # ==========================================================
    for item in QUERY_SET:

        qid = item["qid"]
        query = item["query"]

        # Baseline (no retrieval)
        baseline_answer = generate_answer(query)

        # Vanilla RAG (always retrieve)
        passages = retriever.retrieve(query, k=TOP_K)
        vanilla_rag_answer = generate_answer(query, passages)

        instability = semantic_instability(
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

        print(f"[PASS1] {qid} done")

    # ==========================================================
    # Adaptive Gate Threshold (Calibration Split)
    # ==========================================================
    instabilities = np.array(
        [r["semantic_instability"] for r in raw_results]
    )

    split_idx = int(len(instabilities) * CALIBRATION_SPLIT)

    calibration_instability = instabilities[:split_idx]

    gate_threshold = float(
        np.percentile(calibration_instability, GATE_PERCENTILE)
    )

    print(f"[INFO] Gate percentile: {GATE_PERCENTILE}")
    print(f"[INFO] Gate threshold: {gate_threshold:.6f}")

    # ==========================================================
    # PASS 2 — Apply Failure-Aware Gate
    # ==========================================================
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

            # --- answers ---
            "baseline_answer": r["baseline_answer"],
            "vanilla_rag_answer": r["vanilla_rag_answer"],
            "failure_aware_answer": failure_aware_answer,

            # --- gate ---
            "used_retrieval": use_retrieval,
            "decision_explanation": explanation
        })

        print(f"[PASS2] {r['qid']} | retrieve={use_retrieval}")

    # ==========================================================
    # Save
    # ==========================================================
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"[DONE] Saved results → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
