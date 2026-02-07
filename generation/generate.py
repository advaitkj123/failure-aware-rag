import torch
from .model import tokenizer, model, GEN_KWARGS
from .prompts import prompt_no_retrieval, prompt_with_retrieval


def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_KWARGS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_pair(query, passages):
    answer_no_rag = generate_answer(prompt_no_retrieval(query))
    answer_rag = generate_answer(prompt_with_retrieval(query, passages))
    return answer_no_rag, answer_rag
