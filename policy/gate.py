def should_retrieve(semantic_instability: float,
                    threshold: float) -> bool:
    """
    Failure-aware retrieval gate.

    Returns:
        True  -> allow retrieval
        False -> skip retrieval (fallback to baseline)
    """
    # Retrieve ONLY if instability is BELOW threshold
    return semantic_instability < threshold
