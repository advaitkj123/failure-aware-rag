import torch
from .model import tokenizer, model, GEN_KWARGS
from .prompts import prompt_no_retrieval, prompt_with_retrieval


def _extract_answer(full_text: str) -> str:
    """
    Removes prompt text and returns only the generated answer.
    """
    if "[/INST]" in full_text:
        return full_text.split("[/INST]")[-1].strip()
    return full_text.strip()


def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_answer(query: str, passages=None):
    if passages:
        context = "\n".join(passages)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"

    # existing generation code
    return generate_text(prompt)



def generate_pair(query: str, passages: list[str]):
    answer_no_rag = generate_answer(
        prompt_no_retrieval(query)
    )
    answer_rag = generate_answer(
        prompt_with_retrieval(query, passages)
    )
    return answer_no_rag, answer_rag
