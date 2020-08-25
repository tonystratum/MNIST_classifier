import numpy as np


# activation functions and their derivatives
# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def sigmoid_d(x):
    return np.exp(-x) / np.power(1 + np.exp(-x), 2.)


# ReLU
def relu(x):
    return np.maximum(x, 0)


# ReLU derivative
def relu_d(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# tanh
def tanh(x):
    return np.tanh(x)


# tanh derivative
def tanh_d(x):
    return np.power(np.cosh(x), -2.)


# softmax
def softmax(x):
    exp_max = np.exp(x - np.max(x, axis=0, keepdims=True))
    out = exp_max / np.sum(exp_max, axis=0, keepdims=True)
    return out


def activation_selector(activation, last=False):
    """
    Activation function selector. User passes the parameter "activation" and gets the function of choice.
    :param activation: string with the name of the function
    :param last: flag to check the last layer.
    :return: tuple of a function and its derivative, according to the param "activation".
    """
    if activation is not None:
        if activation == 'relu':
            return relu, relu_d
        elif activation == 'sigmoid':
            return sigmoid, sigmoid_d
        elif activation == 'tanh':
            return tanh, tanh_d
        elif activation == 'softmax':
            assert last
            return softmax, None
            # softmax requires no derivative,
            # as it is the last layer,
            # and the dZ is already computed
            # according to the cost function
            # so return softmax and None
    return None, None
