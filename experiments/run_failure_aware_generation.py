# experiments/run_failure_aware_generation.py

import json
import numpy as np
from pathlib import Path

from generation.generate import generate_answer
from retrieval.bm25_retriever import build_retriever
from features.answer_instability import (
    semantic_instability,
    logical_instability
)
from policy.gate import should_retrieve
from policy.explain_gate import explain_decision


# ==========================================================
# CONFIG
# ==========================================================
QUERY_PATH = "data/processed/query_set_200.json"
OUTPUT_PATH = Path("results/failure_aware_outputs.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GATE_PERCENTILE = 75
CALIBRATION_SPLIT = 0.7
TOP_K = 2


# ==========================================================
# Load Query Set
# ==========================================================
with open(QUERY_PATH, "r", encoding="utf-8") as f:
    QUERY_SET = json.load(f)


# ==========================================================
# Main
# ==========================================================
def main():

    retriever = build_retriever()

    raw_results = []

    print(f"[INFO] Loaded {len(QUERY_SET)} queries")

    # ======================================================
    # PASS 1 — Baseline + Vanilla RAG + Instability
    # ======================================================
    for item in QUERY_SET:

        qid = item["qid"]

        # -------- CLEAN QUERY --------
        if isinstance(item["query"], dict):
            query = item["query"].get("text", "")
        else:
            query = item["query"]

        query = str(query).strip()

        # -------- Baseline --------
        baseline_answer = generate_answer(query)

        # -------- Vanilla RAG --------
        passages = retriever.retrieve(query, k=TOP_K)
        vanilla_rag_answer = generate_answer(query, passages)

        # -------- Instability --------
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

    # ======================================================
    # Adaptive Gate (Calibration Split)
    # ======================================================
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

    # ======================================================
    # PASS 2 — Failure-Aware Selection + NLI Scoring
    # ======================================================
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

        # -------- NLI Scoring --------
        vanilla_logic = logical_instability(
            r["baseline_answer"],
            r["vanilla_rag_answer"]
        )

        failure_logic = logical_instability(
            r["baseline_answer"],
            failure_aware_answer
        )

        final_results.append({
            "qid": r["qid"],
            "query": r["query"],

            # instability
            "semantic_instability": r["semantic_instability"],

            # answers
            "baseline_answer": r["baseline_answer"],
            "vanilla_rag_answer": r["vanilla_rag_answer"],
            "failure_aware_answer": failure_aware_answer,

            # gate
            "used_retrieval": use_retrieval,
            "decision_explanation": explanation,

            # NLI (vanilla)
            "vanilla_entailment": vanilla_logic["entailment"],
            "vanilla_neutral": vanilla_logic["neutral"],
            "vanilla_contradiction": vanilla_logic["contradiction"],

            # NLI (failure-aware)
            "failure_entailment": failure_logic["entailment"],
            "failure_neutral": failure_logic["neutral"],
            "failure_contradiction": failure_logic["contradiction"],
        })

        print(f"[PASS2] {r['qid']} | retrieve={use_retrieval}")

    # ======================================================
    # Save Results
    # ======================================================
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"[DONE] Saved results → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()