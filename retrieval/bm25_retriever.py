import os
import pickle
from rank_bm25 import BM25Okapi
from datasets import load_dataset

def load_wiki_corpus(limit=500):
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    texts = []
    for ex in dataset:
        if "text" in ex and len(ex["text"]) > 200:
            texts.append(ex["text"][:1000])
        if len(texts) >= limit:
            break

    return texts


# -----------------------------
# BM25 Retriever
# -----------------------------
class BM25Retriever:
    def __init__(self, corpus_texts):
        """
        corpus_texts: List[str]
        """
        self.texts = corpus_texts

        # Fast, standard BM25 tokenization (used in IR literature)
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 3):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.texts[i] for i in top_idx]


# -----------------------------
# Convenience Builder
# -----------------------------
def build_retriever(limit: int = DEFAULT_LIMIT) -> BM25Retriever:
    corpus = load_wiki_corpus(limit=limit)
    return BM25Retriever(corpus)
