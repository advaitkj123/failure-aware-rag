import numpy as np

def compute_percentile_threshold(instability_scores, percentile=75):
    return float(np.percentile(instability_scores, percentile))

def should_retrieve(instability, threshold):
    return instability > threshold
