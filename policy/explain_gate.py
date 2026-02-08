
def explain_decision(instability: float, tau: float = 0.18) -> str:
    """
    Explain the failure-aware retrieval decision based on semantic instability.

    Parameters
    ----------
    instability : float
        Measured semantic instability between vanilla and retrieved answers.
    tau : float
        Decision threshold for triggering retrieval.

    Returns
    -------
    str
        Human-readable explanation of the decision.
    """
    if instability >= tau:
        return (
            f"Retrieval was triggered because semantic instability "
            f"({instability:.3f}) exceeded the threshold ({tau:.2f})."
        )
    else:
        return (
            f"Retrieval was skipped because semantic instability "
            f"({instability:.3f}) remained below the threshold ({tau:.2f})."
        )
