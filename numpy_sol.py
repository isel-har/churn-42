import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        pass


class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0, beta_2=0, epsilon=0):
        pass


class Dropout:
    ...

class EarlyStopping:
    ...



class NDense:
    def __init__(self):
        self.size = 0
        self.input_shape = 0
        self.activation = None

        self.weights = None
        self.biases  = None

        self.weights_gradients = None
        self.biases_gradients  = None

        self.output_cache = None
        self.input_cache = None
        self.z_cache = None

    def forward(self, X):
        ...

    def backward(self):
        ...



class NNet:

    def __init__(self, layers=None):
        self.optimizer = None
        self.layers    = layers


    def compile(self, optimizer=None, metrics=['AUC']):

        for layer in self.layers:
            ...


    def add(self, layer):
        ...

    def fit(self, X, y, epochs=100):
        ...
    
    def predict(self, X):
        ...
