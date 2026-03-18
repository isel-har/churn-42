from scipy.special import expit
import numpy as np


class Relu:

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def __call__(self, inputs, derivative=False):
        if not derivative:
            return np.maximum(self.threshold, inputs)

        return (inputs > self.threshold).astype(inputs.dtype)


class Sigmoid:

    def __init__(self):
        pass

    def __call__(self, inputs, derivative=False):
        
        x = np.asarray(inputs, dtype=float)
        s = expit(x)
        return s if not derivative else s * (1.0 - s)

