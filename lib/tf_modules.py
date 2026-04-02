import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        super().__init__(name=name)
        # Xavier Initialization
        limit = tf.sqrt(6 / (in_features + out_features))
        w_init = tf.random.uniform([in_features, out_features], minval=-limit, maxval=limit)
        
        self.w = tf.Variable(w_init, name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        self.activation = activation
    
    def __call__(self, X):
        y = tf.matmul(X, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y



class TNet(tf.Module):

    def __init__(self, layers=[], name=None):
        super().__init__(name=name)
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_loss(self, y_true, y_pred):

        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        term_1 = y_true * tf.math.log(y_pred)
        term_0 = (1 - y_true) * tf.math.log(1 - y_pred)
        
        return -tf.reduce_mean(term_1 + term_0)


    def fit(self, x, y, epochs=10, learning_rate=0.01, batch_size=32):
    # 1. Prepare the target shape
        y = tf.reshape(y, (-1, 1))
        
        # 2. Use tf.data for efficient batching and shuffling
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            # 3. Iterate over the batches
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    y_pred = self(x_batch)
                    loss = self.compute_loss(y_batch, y_pred)

                gradients = tape.gradient(loss, self.trainable_variables)
                
                for var, grad in zip(self.trainable_variables, gradients):
                    var.assign_sub(learning_rate * grad)
                
                epoch_loss_avg.update_state(loss)

            print(f"Epoch {epoch}, Average Loss: {epoch_loss_avg.result():.4f}")


    def predict(self, x):
        return self(x)

    def predict_classes(self, x, threshold=0.5):
        probs = self.predict(x)
        return tf.cast(probs > threshold, tf.float32)
    
