import tensorflow as tf
import numpy as np
from scipy.stats import norm


class Net(tf.keras.Model):
    def __init__(self, state, hidden_units, actions_n):
        super(Net, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state,))
        self.hiidens = []
        for units in hidden_units:
            print(f'units = {units}')
            layer = tf.keras.layers.Dense(units, activation='relu', kernel_initializer='GlorotNormal')
            self.hiidens.append(layer)
        self.output_layer = tf.keras.layers.Dense(actions_n, activation='linear', kernel_initializer='RandomNormal')

    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hiidens:
            z = layer(z)
        return self.output_layer(z)


if __name__ == "__main__":
    nor = norm(0,1)
    N = 100
    X = np.zeros(shape=(N,2))
    X[:, 0] = nor.rvs(N)
    X[:, 1] = nor.rvs(N)
    print(X.shape)

    net = Net(2, [10,10], 2)

    pred = net(X[:])
    print(pred.shape)





