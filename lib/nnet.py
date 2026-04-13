from lib.layers import DenseLayer, Dropout
import numpy as np
from tqdm import tqdm

class NNet:

    def __init__(
        self,
        layers=None,
        optimizer=None,
        earlystopping=None,
        loss=None,
    ):

        self.optimizer     = optimizer
        self.earlystopping = earlystopping
        self.loss          = loss
        self.layers        = list()
        self.earlystopping_bool = False
        self.class_weight   = None

        if layers is not None:
            for layer in layers:
                self.add(layer)


    def add(self, layer):

        if isinstance(layer, DenseLayer):
            if layer.output_size == 0:
                raise Exception("output size not passed or equal to 0")

            shape = [layer.input_size, layer.output_size]

            if layer.input_size == 0:
                dense_len = len([l for l in self.layers if isinstance(l, DenseLayer)])
                if dense_len > 0:

                    for layer_ in reversed(self.layers):
                        if hasattr(layer_, "output_size"):
                            layer.input_size = layer_.output_size
                            break

                    shape[0] = layer.input_size
                else:
                    raise Exception("first layer must have input size")

            layer.biases  = DenseLayer.initializers_dict['constant'](layer.bias_initializer, (1, shape[1]))
            layer.weights = DenseLayer.initializers_dict[layer.weights_initializer](tuple(shape))

            layer.weights_gradients = np.zeros(shape=(shape[0], shape[1]), dtype=np.float32)
            layer.biases_gradients = np.zeros(shape=(1, shape[1]), dtype=np.float32)


        elif not isinstance(layer, Dropout):
            raise Exception("Layer instance must be a dense or dropout layer.")
        
        self.layers.append(layer)



    def set_weights(self, weights=None):
    
        i = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights = weights[i]['w']
                layer.biases  = weights[i]['b']
                i += 1

    def get_weights(self):

        weights = []
        for layer in self.layers:
            if isinstance(layer, Dropout):
                continue
            weights.append({
                'w': layer.weights.copy(),
                'b': layer.biases.copy()
            })
        return weights


    def forward(self, inputs):
        
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def backward(self, probs, ybatch):
        dout = self.loss.backward(probs, ybatch)

        if self.class_weight is not None:
            weights = np.ones_like(ybatch, dtype=float)
            for class_val, weight in enumerate(self.class_weight):
                weights[ybatch == class_val] = weight
            
            dout = dout * weights

        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        self.optimizer.update_step(self.layers)


    def data_handler(self, X=None, y=None):

        if X is None or y is None:
            raise Exception("X or y cannot be None")
        
        if len(X) != len(y):
            raise Exception("Length of X an y should be equal.")
        

        def check_np_array(obj):
            if not isinstance(obj, np.ndarray):

                if callable(obj.to_numpy):
                    np_obj = obj.to_numpy()
                    if len(obj.shape) == 1:
                        np_obj = np_obj.reshape(-1, 1)
                    return np_obj
                else:
                    raise Exception("numpy array object required")

            return obj

        X_np = check_np_array(X)
        y_np = check_np_array(y)

        return X_np, y_np


    def check_validation_data(self, validation_data):
        
        if validation_data is None:
            raise Exception("validation dataset required for earlystopping.")
        if not isinstance(validation_data, tuple):
            raise Exception("validation dataset must be a tuple (x, y).")
 
        self.earlystopping_bool = True
        return self.data_handler(validation_data[0], validation_data[1])


    def fit(
        self,
        X=None,
        y=None,
        epochs=1,
        batch_size=16,
        validation_data=None,
        class_weight=None
    ):
        self.class_weight = class_weight
        if self.earlystopping is not None:

            self.X_val, self.y_val = self.check_validation_data(validation_data)
        X_np, y_np   = self.data_handler(X, y)

        self.training_mode = True
        rows = X_np.shape[0]

        if callable(self.optimizer.init_hyper_params):
            self.optimizer.init_hyper_params(self.layers)
    
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            if self.earlystopping_bool:
                
                val_probs = self.forward(self.X_val)
                val_loss  = self.loss(val_probs, self.y_val)
                if self.earlystopping(val_loss, self):
                    break

            pbar = tqdm(range(0, rows, batch_size),
                desc="",
                bar_format="{l_bar}{bar}",
                ascii=" >=",
                ncols=100)

            for i in pbar:
 
                end = min(i + batch_size, rows)
         
                xbatch = X_np[i:end]
                ybatch = y_np[i:end].reshape(-1, 1)

                probs = self.forward(xbatch)
                self.backward(probs, ybatch)


        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
    
        return self

    def predict(self, X):
        return self.forward(X)

    # def predict_proba(self, X):
    #     probs = self.forward(X)

    #     if probs.ndim == 1:
    #         probs = probs.reshape(-1, 1)

    #     return np.hstack([1 - probs, probs])

    