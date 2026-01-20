
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

class GradientDescent(Optimizer):

    def __call__(self, loss, layers):

        for layer in reversed(layers):
            dloss = layer.backward(loss)

            # update
        for layer in layers:
            layer.weights -= self.learning_rate * layer.weights_gradients
            layer.biases  -= self.learning_rate * layer.biases_gradients