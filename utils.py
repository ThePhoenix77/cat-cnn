import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def bce(y, y_hat):
    epsilon = 1e-15                                                         #smallest possible value close to 0, to avoid it
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)                            #np.clip to replace all values minimum to epsilon with epsilon and vice versa with values over (1 - epsilon)
    return -(np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))      #np.mean for arithmetic average
