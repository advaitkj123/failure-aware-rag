from sentence_transformers import SentenceTransformer
import numpy as np

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_instability(ans_no_rag: str, ans_rag: str) -> float:
    """
    Cosine distance between answer embeddings.
    Higher = more semantic drift.
    """
    emb = _embedder.encode([ans_no_rag, ans_rag], normalize_embeddings=True)
    cosine_sim = float(np.dot(emb[0], emb[1]))
    return 1.0 - cosine_sim


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_NLI_MODEL = "roberta-large-mnli"
_nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL)
_nli_model.eval()

def logical_instability(ans_no_rag: str, ans_rag: str) -> dict:
    """
    NLI from no-RAG -> RAG answer.
    Returns entailment / neutral / contradiction probabilities.
    """
    inputs = _nli_tokenizer(
        ans_no_rag,
        ans_rag,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        logits = _nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return {
        "entailment": float(probs[2]),
        "neutral": float(probs[1]),
        "contradiction": float(probs[0]),
    }


def structural_drift(ans_no_rag: str, ans_rag: str) -> dict:
    """
    Simple surface-level drift metrics.
    """
    len_no = len(ans_no_rag.split())
    len_rag = len(ans_rag.split())

    return {
        "length_diff": abs(len_rag - len_no),
        "length_ratio": (len_rag + 1e-5) / (len_no + 1e-5),
    }
