
import tensorflow as tf

class TLayer(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        super().__init__(name=name)
        self.activation = activation
        # Initialize weights (random normal) and biases (zeros)
        w_init = tf.random.normal([in_features, out_features], stddev=0.1)
        self.w = tf.Variable(w_init, name='w')
        
        b_init = tf.zeros([out_features])
        self.b = tf.Variable(b_init, name='b')

    def __call__(self, x):
        # x is expected to be shape (Batch_Size, In_Features)
        y = tf.matmul(x, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y

class TSequential(tf.Module):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_loss(self, y_true, y_pred):
        # 1. Clip values to prevent log(0) which produces NaNs
        # We limit predictions to range [1e-7, 0.9999999]
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 2. Calculate Binary Cross Entropy Formula
        # Formula: - (y * log(p) + (1-y) * log(1-p))
        term_1 = y_true * tf.math.log(y_pred)
        term_0 = (1 - y_true) * tf.math.log(1 - y_pred)
        
        return -tf.reduce_mean(term_1 + term_0)


    def fit(self, x, y, epochs=10, learning_rate=0.01):
        # Ensure data is float32
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        y = tf.reshape(y, (-1, 1))
        print("Starting training...")
        for epoch in range(epochs):
            
            with tf.GradientTape() as tape:
                # FIXED: Pass the WHOLE batch (x) at once, not row-by-row.
                # This is much faster and ensures matrix shapes match.
                y_pred = self(x)
                loss    = self.compute_loss(y, y_pred)

            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            
            # Update weights
            for var, grad in zip(self.trainable_variables, gradients):
                var.assign_sub(learning_rate * grad)

            # Print status
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")


    def predict(self, x):
        # Ensure input is float32
        x = tf.cast(x, dtype=tf.float32)
        return self(x)


    def predict_classes(self, x, threshold=0.5):
        """ Returns 0.0 or 1.0 based on threshold """
        probs = self.predict(x)
        # Return 1.0 if prob > threshold, else 0.0
        return tf.cast(probs > threshold, tf.float32)