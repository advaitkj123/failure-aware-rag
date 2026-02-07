def prompt_no_retrieval(query: str) -> str:
    return f"""[INST]
Answer the following question as accurately and concisely as possible.

Question:
{query}
[/INST]"""


def prompt_with_retrieval(query: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    return f"""[INST]
You are given some background information. Use it if relevant, but do not assume it is always correct.

Background:
{context}

Question:
{query}
[/INST]"""
