def prompt_no_retrieval(query: str) -> str:
    return f"""[INST]
Answer the following question as accurately and concisely as possible.
If you are unsure, say you do not know.

Question:
{query}
[/INST]"""

def prompt_with_retrieval(query: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    return f"""[INST]
You are given some background information.
Use it only if it is relevant and consistent.
Do not attempt to reconcile conflicting information.

Background:
{context}

Question:
{query}
[/INST]"""
