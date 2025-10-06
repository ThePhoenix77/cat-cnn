import numpy as np
from cnn_layers import conv2d, max_pooling, conv2d_backward, max_pool_backward
from cnn_network import Dense
from utils import sigmoid, sigmoid_prime


class CatCNN:
    def __init__(self, input_shape=(64, 64)):
        self.kernel1 = np.random.randn(3, 3) * 0.01
        self.kernel2 = np.random.randn(3, 3) * 0.01

        h, w = input_shape
        h, w = h - 3 + 1, w - 3 + 1  # conv1
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1  # pool1
        h, w = h - 3 + 1, w - 3 + 1  # conv2
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1  # pool2

        dense_input_size = h * w
        self.dense = Dense(dense_input_size, 1)

    def forward(self, x):
        # Layer 1
        self.conv1 = conv2d(x, self.kernel1)
        self.pool1 = max_pooling(self.conv1)

        # Layer 2
        self.conv2 = conv2d(self.pool1, self.kernel2)
        self.pool2 = max_pooling(self.conv2)

        self.flatten = self.pool2.flatten().reshape(-1, 1)
        self.output = sigmoid(self.dense.forward(self.flatten))
        return self.output

    def backward(self, y_true, x):
        error = self.output - y_true
        d_output = error * sigmoid_prime(self.dense.y)

        dL_dx_dense, grads_dense = self.dense.backward(d_output)
        dL_dx_dense = dL_dx_dense.reshape(self.pool2.shape)

        # Layer 2 backward
        dL_dpool2 = max_pool_backward(dL_dx_dense, self.conv2)
        dL_dx_conv2, d_kernel2 = conv2d_backward(dL_dpool2, self.pool1, self.kernel2)

        # Layer 1 backward
        dL_dpool1 = max_pool_backward(dL_dx_conv2, self.conv1)
        _, d_kernel1 = conv2d_backward(dL_dpool1, x, self.kernel1)

        return {
            "kernel1": d_kernel1,
            "kernel2": d_kernel2,
            "dense_W": grads_dense["W"],
            "dense_b": grads_dense["b"]
        }
