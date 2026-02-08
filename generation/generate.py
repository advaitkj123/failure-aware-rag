
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16
)

model.eval()


def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_answer(query: str, passages=None) -> str:
    if passages:
        context = "\n".join(passages[:2])  # GPU-safe context limit
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"

    return generate_text(prompt)
