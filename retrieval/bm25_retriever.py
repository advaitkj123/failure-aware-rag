# retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi

STATIC_CORPUS = [
    "Alexander Fleming discovered penicillin in 1928.",
    "Canberra is the capital city of Australia.",
    "William Shakespeare wrote Hamlet.",
    "The Titanic sank in 1912.",
    "Pluto is classified as a dwarf planet.",
    "Alexander Graham Bell invented the telephone.",
    "Water boils at 100 degrees Celsius.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Jupiter is the largest planet in the Solar System.",
    "George Washington was the first President of the United States."
]

class BM25Retriever:
    def __init__(self, texts):
        self.texts = texts
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=2):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.texts[i] for i in top_idx]

def build_retriever():
    return BM25Retriever(STATIC_CORPUS)
