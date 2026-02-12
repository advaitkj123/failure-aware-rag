# retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi
from datasets import load_dataset
import random

MAX_DOCS = 2000  # publication-grade but safe


def load_wiki_corpus(limit=MAX_DOCS):
    """
    Stream Wikipedia and collect limited number of documents.
    No disk caching. No explosion.
    """

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    texts = []
    for ex in dataset:
        if "text" in ex and len(ex["text"]) > 300:
            texts.append(ex["text"][:1000])

        if len(texts) >= limit:
            break

    return texts


class BM25Retriever:

    def __init__(self, corpus_texts):
        self.texts = corpus_texts
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 2):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.texts[i] for i in top_idx]


def build_retriever():
    corpus = load_wiki_corpus()
    return BM25Retriever(corpus)
