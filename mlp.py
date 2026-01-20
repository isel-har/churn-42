from optimizers import Optimizer
from layers import Layer
import numpy as np

class MSequential:

    def __init__(self, layers:list=[]):
        self.layers:list  = layers
        self.compiled  = False
        self.optimizer = None
        self.metrics   = []

    def compile(self, optimizer=None, metrics=[]):

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise Exception(f'optimizer not instance of Optimizer class.')

        if len(self.layers) < 2:
            raise Exception('invalid layer size (at least one hidden and output layer) required.')

        self.metrics   = metrics
        self.optimizer = optimizer
        self.compiled  = True
        print("compiled")


    def train(self, x, y):
        for xi, yi in zip(x, y):

            xi = xi.reshape(1, -1)
            yi = yi.reshape(1, -1)

            p = xi
            for layer in self.layers:
                p = layer.forward(p)

            dloss = p - yi
            self.optimizer(dloss, self.layers)


    def fit(self, x, y, epochs=10):
        if not self.compiled:
            raise Exception('run compile function before training.')

        for e in range(epochs):
            self.train(x, y)
            print(f"epoch: {e}")



    def add(self, layer:Layer):
        self.layers.append(layer)


    def remove(self, index:int):
    
        if index >= 0 and index < len(self.layers):
            self.layers.pop(index)


    def predict_proba(self, x):
        """
        Returns the raw output of the last layer (probabilities/activations).
        Optimized to handle the entire batch 'x' at once.
        """
        # Start with the input matrix
        # If x is (samples, features), ensure weights are shaped correctly
        r = x
        
        for layer in self.layers:
            # Vectorized forward pass: (batch @ weights) + bias
            # This assumes layer.weights is (input_dim, output_dim)
            z = r @ layer.weights + layer.biases
            r = layer.activation(z)
            
        return r

    def predict(self, x):
        """
        Returns the class with the highest probability.
        """
        # 1. Get the raw probabilities
        probabilities = self.predict_proba(x)
        
        # 2. Find the index of the maximum value along the last axis
        # axis=1 corresponds to the columns (the classes)
        return np.argmax(probabilities, axis=1)

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)


