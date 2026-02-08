import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# DEVICE (GPU-SAFE)
# -----------------------------
_DEVICE = torch.device("cpu")


# -----------------------------
# Semantic Instability (Embedding Drift)
# -----------------------------
_embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=_DEVICE
)

def semantic_instability(baseline_answer: str,
                         vanilla_rag_answer: str) -> float:
    """
    Cosine distance between answer embeddings.
    Range: [0, 2] (practically ~[0, 0.6])
    Higher = more semantic drift.
    """
    emb = _embedder.encode(
        [baseline_answer, vanilla_rag_answer],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    cosine_sim = float(np.dot(emb[0], emb[1]))
    return 1.0 - cosine_sim


# -----------------------------
# Logical Instability (NLI)
# -----------------------------
_NLI_MODEL = "roberta-large-mnli"

_nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(
    _NLI_MODEL
).to(_DEVICE)
_nli_model.eval()

def logical_instability(baseline_answer: str,
                        vanilla_rag_answer: str) -> dict:
    """
    NLI from baseline -> vanilla RAG answer.
    Returns entailment / neutral / contradiction probabilities.
    """
    inputs = _nli_tokenizer(
        baseline_answer,
        vanilla_rag_answer,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return {
        "entailment": float(probs[2]),
        "neutral": float(probs[1]),
        "contradiction": float(probs[0]),
    }


# -----------------------------
# Structural Drift (Surface-Level)
# -----------------------------
def structural_drift(baseline_answer: str,
                     vanilla_rag_answer: str) -> dict:
    """
    Surface-level drift metrics (non-semantic).
    """
    len_base = len(baseline_answer.split())
    len_rag = len(vanilla_rag_answer.split())

    return {
        "length_diff": abs(len_rag - len_base),
        "length_ratio": (len_rag + 1e-6) / (len_base + 1e-6),
    }
