def should_retrieve(instability: float, threshold: float = 0.3) -> bool:
    return instability > threshold