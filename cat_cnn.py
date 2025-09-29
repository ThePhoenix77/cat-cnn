import numpy as np
from cnn_layers import conv2d, max_pooling
from cnn_network import Dense
from utils import sigmoid, sigmoid_prime

class CatCNN:
    def __init__(self):
        self.kernel = np.random.randn(3, 3) * 0.01
        self.dense = Dense(1024, 1)

    def forward(self, x):
        self.conv = conv2d(x, self.kernel)
        self.pool = max_pooling(self.conv)
        self.flatten = self.pool.flatten().reshape(-1, 1)
        self.output = sigmoid(self.dense.forward(self.flatten))
        return self.output

    def backward(self, y_true, x):
        error = self.output - y_true
        d_output = error * sigmoid_prime(self.dense.y)
        dL_dx, grads = self.dense.backward(d_output)
        dL_dx = dL_dx.reshape(self.pool.shape)
        from cnn_layers import max_pool_backward, conv2d_backward
        dL_pool = max_pool_backward(dL_dx, self.conv)
        _, d_kernel = conv2d_backward(dL_pool, x, self.kernel)
        return {"kernel": d_kernel, "dense_W": grads["W"], "dense_b": grads["b"]}
