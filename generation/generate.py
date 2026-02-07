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


def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_KWARGS)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _extract_answer(decoded)


def generate_pair(query: str, passages: list[str]):
    answer_no_rag = generate_answer(
        prompt_no_retrieval(query)
    )
    answer_rag = generate_answer(
        prompt_with_retrieval(query, passages)
    )
    return answer_no_rag, answer_rag
