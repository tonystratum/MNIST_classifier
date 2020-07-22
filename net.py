import numpy as np

from load_cached_weights import load_weights


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
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


# function to choose between the activation functions in place
def activation_function(Z, name, d=False):
    if d:
        if name == 'relu':
            return relu_d(Z)
        elif name == 'sigmoid':
            return sigmoid_d(Z)
        elif name == 'tanh':
            return tanh_d(Z)
    else:
        if name == 'relu':
            return relu(Z)
        elif name == 'sigmoid':
            return sigmoid(Z)
        elif name == 'tanh':
            return tanh(Z)
        elif name == 'softmax':
            return softmax(Z)


def forward(X, weights, biases, activations):
    A = [X]
    for weight, bias, activation in zip(weights, biases, activations):
        Z = np.dot(weight, A[-1]) + bias
        A.append(activation_function(Z, activation))

    return A


# load cached weights (if available)

def predict(X, cached_weights_path):
    cached_weights = load_weights(cached_weights_path)
    weights = [np.array(cached_weights['W'][W]) for W in sorted(cached_weights['W'].keys())]
    biases = [np.array(cached_weights['b'][b]) for b in sorted(cached_weights['b'].keys())]

    # set the activations
    activations = [
        'relu',
        'softmax',
    ]

    # predict it
    prediction = forward(X, weights, biases, activations)[-1]
    return prediction
