import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        self.epsilon = 1e-15  # prevent log(0)

    def __call__(self, probs, ybatch):
        probs = np.clip(probs, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(
            ybatch * np.log(probs) +
            (1 - ybatch) * np.log(1 - probs)
        )
        return loss

    def backward(self, probs, ybatch):
        # The derivative of BCE with respect to probabilities (dLoss/dProbs)
        # Formula: -(y/p) + (1-y)/(1-p)
        probs = np.clip(probs, self.epsilon, 1 - self.epsilon)
        return (-(ybatch / probs) + (1 - ybatch) / (1 - probs)) / ybatch.shape[0]