import numpy as np
# import activations as av

class Layer:

    def __init__(self, lsize, xshape, activation, seed=42):
        # lsize  = output dimension
        # xshape = input dimension

        self.activation = activation
        self.rng        = np.random.default_rng(seed=seed)
        self.xshape     = xshape
        self.lsize      = lsize

        # biases must be shape (1 , out_dim)
        self.biases = np.zeros((1, lsize))

        # ---- FIX: correct weight shape ----
        # Xavier initialization for sigmoid
        limit = np.sqrt(1.0 / xshape)
        self.weights = self.rng.uniform(
            -limit,
            limit,
            size=(xshape, lsize)   # <<< FIX SHAPE (in_dim , out_dim)
        )

        # gradients
        self.weights_gradients = None
        self.biases_gradients  = None

        # caching
        self.output_cache = None
        self.input_cache  = None
        self.z_cache      = None


    def forward(self, x):
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        self.input_cache = x
        self.z_cache = x @ self.weights + self.biases
        self.output_cache = self.activation(self.z_cache)
        return self.output_cache

    def backward(self, dl_out):
        # if dl_out.ndim == 1:
        #     dl_out = dl_out.reshape(1, -1)
        dz = dl_out * self.activation(self.z_cache, der=True)
        self.weights_gradients = self.input_cache.T @ dz
        self.biases_gradients  = dz.sum(axis=0, keepdims=True)
        return dz @ self.weights.T



