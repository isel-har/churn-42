import numpy as np
from scipy.special import expit

def sigmoid(x, der=False):
    x = np.asarray(x, dtype=float)
    s = expit(x)
    return s if not der else s * (1.0 - s)

def relu(x, der=False):
    x = np.asarray(x)
    return (x > 0).astype(x.dtype) if der else np.maximum(0, x)
