import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def update_step(self, layers):
        for layer in layers:
            if not hasattr(layer, "weights"):
                continue
            layer.weights -= self.learning_rate * layer.weights_gradients
            layer.biases  -= self.learning_rate * layer.biases_gradients


class Adam:
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        layers=None
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.built_params = False

        if layers:
            self.init_hyper_params(layers)

    def init_hyper_params(self, layers):
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []

        for layer in layers:
            if not hasattr(layer, "weights"):
                continue
            self.m_w.append(np.zeros_like(layer.weights))
            self.v_w.append(np.zeros_like(layer.weights))
            self.m_b.append(np.zeros_like(layer.biases))
            self.v_b.append(np.zeros_like(layer.biases))
        self.built_params = True


    def update_step(self, layers):
        self.t += 1
        param_i = 0

        for layer in layers:
            if not hasattr(layer, "weights"):
                continue

            # Update moments
            self.m_w[param_i] = (self.beta_1 * self.m_w[param_i]) + (1 - self.beta_1) * layer.weights_gradients
            self.v_w[param_i] = (self.beta_2 * self.v_w[param_i]) + (1 - self.beta_2) * (layer.weights_gradients ** 2)

            self.m_b[param_i] = (self.beta_1 * self.m_b[param_i]) + (1 - self.beta_1) * layer.biases_gradients
            self.v_b[param_i] = (self.beta_2 * self.v_b[param_i]) + (1 - self.beta_2) * (layer.biases_gradients ** 2)

            # Bias correction
            m_w_hat = self.m_w[param_i] / (1 - self.beta_1 ** self.t)
            v_w_hat = self.v_w[param_i] / (1 - self.beta_2 ** self.t)

            m_b_hat = self.m_b[param_i] / (1 - self.beta_1 ** self.t)
            v_b_hat = self.v_b[param_i] / (1 - self.beta_2 ** self.t)

            # Update params
            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.biases  -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            param_i += 1