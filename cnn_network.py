import numpy as np

#we create a class here so each instance will have its own weights and biases
class Dense:
    def __init__(self, n_inputs, n_outputs):                        #constructor
        self.W = np.random.randn(n_outputs, n_inputs) * 0.01        #initializing the weights with small random numbers
        self.b = np.zeros((n_outputs, 1))                           #bias vector

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.W, x) + self.b                         # y = W.x + b
        return self.y

    def backward(self, dL_dy):
        dW = np.dot(dL_dy, self.x.T)
        db = dL_dy
        dL_dx = np.dot(self.W.T, dL_dy)
        return dL_dx, {"W": dW, "b": db}

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

