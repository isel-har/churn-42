from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import sys


class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):

        super().__init__(name=name)
        w_init = tf.random.normal([in_features, out_features], stddev=0.1)
        self.w = tf.Variable(w_init, name='w')
        

        b_init = tf.zeros([out_features])
        self.b = tf.Variable(b_init, name='b')
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


    def fit(self, x, y, epochs=10, learning_rate=0.01):

        y = tf.reshape(y, (-1, 1))

        for epoch in range(epochs):
            
            with tf.GradientTape() as tape:
                # FIXED: Pass the WHOLE batch (x) at once, not row-by-row.
                # This is much faster and ensures matrix shapes match.
                y_pred  = self(x)
                loss    = self.compute_loss(y, y_pred)

            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            
            # Update weights
            for var, grad in zip(self.trainable_variables, gradients):
                var.assign_sub(learning_rate * grad)

            # Print status
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    def predict(self, x):
        return self(x)

    def predict_classes(self, x, threshold=0.5):
        probs = self.predict(x)
        return tf.cast(probs > threshold, tf.float32)
    

def main():

    try:

        transformer = joblib.load("transformer.pkl")
        train_df = pd.read_csv(sys.argv[1])
        y = train_df['TARGET'].copy()
        train_df = train_df.drop(columns=['TARGET'])

        transformed_df = transformer.transform(train_df).astype(np.float32)

        y = y.astype(np.float32)

        del train_df


        tnn = TNet(layers=[
            Dense(in_features=transformed_df.shape[1], out_features=128, activation=tf.nn.relu),
            Dense(in_features=128, out_features=64, activation=tf.nn.relu),
            Dense(in_features=64, out_features=1, activation=tf.nn.sigmoid),
        ])

        tnn.fit(transformed_df, y, epochs=100, learning_rate=0.01)

        del transformed_df
        del y

        test_df = pd.read_csv(sys.argv[2])
        y_test = test_df['TARGET']
        X_test = test_df.drop(columns=['TARGET'])
        transformed_test_df = transformer.transform(X_test).astype(np.float32)
        y_test = y_test.astype(np.float32)
        del X_test

        print("____________________________")
        y_pred = tnn.predict(transformed_test_df).numpy().ravel()
        auc = roc_auc_score(y_test, y_pred)
        print("Tensorflow AUC score:", auc)

    except Exception as e:
        print('exception:',str(e))


if __name__ == '__main__':
    main()