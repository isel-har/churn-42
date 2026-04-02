from .initializers import Initializers
import numpy as np

class Dropout:
    def __init__(self, rate=0.3, seed=42):
        self.rate      = rate
        self.keep_prob = 1.0 - rate
        self.mask      = None
        self.training  = True
        self.rng = np.random.default_rng(seed)
    def __call__(self, inputs):
        if not self.training:
            return inputs

        # Always match inputs shape
        self.mask = (self.rng.random(inputs.shape) < self.keep_prob).astype(np.float32)

        return (inputs * self.mask) / self.keep_prob


    def backward(self, grad_output):
        return (grad_output * self.mask) / self.keep_prob


class DenseLayer:

    initializers_dict = {
        'xavier': Initializers.xavier_init,
        'he': Initializers.he_init,
        'constant': Initializers.constant_init,
        'random': Initializers.random_init,
    }

    def __init__(
        self,
        input_size=0,
        output_size=0,
        activation=None,
        weights_initializer='he',
        bias_initializer=0,
    ):
        self.output_size = output_size
        self.input_size  = input_size 
        self.activation  = activation

        self.weights_initializer = weights_initializer
        self.bias_initializer    = bias_initializer
        
        self.weights = None#self.initializers_dict[weights_initializer](self.shape)
        self.biases  = None#self.initializers_dict['constant'](bias_initializer, (1, output_size))

        self.weights_gradients = None
        self.biases_gradients  = None

        self.output_cache      = None
        self.input_cache       = None
        self.z_cache           = None



    def __call__(self, inputs):

        self.input_cache = inputs
        self.z_cache = (inputs @ self.weights) + self.biases
        self.output_cache = self.activation(self.z_cache, False)
        return self.output_cache


    def backward(self, dout):
        dz = dout * self.activation(self.z_cache, True)
        self.weights_gradients = self.input_cache.T @ dz
        self.biases_gradients  = dz.sum(axis=0, keepdims=True)
        return dz @ self.weights.T
