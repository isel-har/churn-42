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
        self.training_mode = False
        self.earlystopping_bool = False

        if layers is not None:
            for layer in layers:
                self.add(layer)


    def add(self, layer: DenseLayer):

        if layer.output_size == 0:
            raise Exception("output size not passed or equal to 0")

        shape = [layer.input_size, layer.output_size]

        if layer.input_size == 0:
            if len(self.layers) > 0:
                layer.input_size = self.layers[-1].output_size
                shape[0] = layer.input_size
            else:
                raise Exception("first layer must have input size")

        layer.biases  = DenseLayer.initializers_dict['constant'](layer.bias_initializer, (1, shape[1]))
        layer.weights = DenseLayer.initializers_dict[layer.weights_initializer](tuple(shape))

        self.layers.append(layer)


    def forward(self, inputs):
        
        for layer in self.layers:

            if not self.training_mode and isinstance(layer, Dropout):
                continue
            inputs = layer(inputs)

        return inputs


    def backward(self, probs, ybatch):
        
        dout = self.loss.backward(probs, ybatch)

        for layer in reversed(self.layers):
    
            if isinstance(layer, Dropout):
                continue
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
        validation_data=None
    ):

        if self.earlystopping is not None:

            self.X_val, self.y_val = self.check_validation_data(validation_data)
        X_np, y_np   = self.data_handler(X, y)

        self.training_mode = True
        rows = X_np.shape[0]

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            if self.earlystopping_bool:
                
                val_probs = self.forward(self.X_val)
                val_loss  = self.loss(val_probs, self.y_val)
                if self.earlystopping(val_loss):
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

        self.training_mode = False
        return self

    def predict(self, X):
        return self.forward(X)

