from generation.generate import generate_pair

if __name__ == "__main__":
    query = "Who wrote the novel 1984?"
    dummy_passages = [
        "George Orwell was an English novelist and journalist.",
        "The novel 1984 is a dystopian social science fiction novel."
    ]

    ans_no_rag, ans_rag = generate_pair(query, dummy_passages)

    print("=== NO RETRIEVAL ===")
    print(ans_no_rag)
    print()

    print("=== WITH RETRIEVAL ===")
    print(ans_rag)
