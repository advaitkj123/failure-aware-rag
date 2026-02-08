
from datasets import load_dataset
from rank_bm25 import BM25Okapi

def load_wiki_corpus(limit=500):
    """
    Stream Wikipedia safely (no full download).
    """
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


class BM25Retriever:
    def __init__(self, corpus_texts):
        self.texts = corpus_texts
        tokenized = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=3):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        return [self.texts[i] for i in top_idx]


def build_retriever(limit=500):
    corpus = load_wiki_corpus(limit)
    return BM25Retriever(corpus)
