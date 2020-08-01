import numpy as np

from activation_functions import activation_selector

class Layer:
    """
    A basic layer model. Types of layers are to be inherited from it.
    :param shape (int): int Shape of this layer.
    :param layer_type (str): Type of the layer, init in the derived class
    :param last (bool): True if the layer is last in the network
    """

    def __init__(self, layer_type: str, shape, activation, last: bool):
        self.layer_type = layer_type
        self.shape = shape
        self.last = last
        self.activation_name = activation

        self.activation, self.activation_derivative = activation_selector(activation, last)


class FullyConnected(Layer):
    """
    Fully connected layer. Child of the Layer class.
    Adds weights (W) and biases (b).
    Both are of numpy.ndarray, shapes of which are determined
    by the shape of the previous layer.
    """

    def __init__(self, shape: int, activation, last: bool = False):
        super().__init__('FullyConnected', shape, activation, last)
        self.W = None
        self.b = None

        self.Z = None
        self.A = None

        self.dA = None
        self.dZ = None

        self.dW = None
        self.db = None

    def initialize(self, previous_shape: int):
        """
        Initialize the layer according to the previous shape.
        To be called in Network.add_layer()
        :param previous_shape: Previous layer shape (int).
        """
        self.W = np.random.randn(self.shape, previous_shape) * .1
        self.b = np.zeros((self.shape, 1))

    def get_shape(self):
        return self.shape

    # TEMP

    def forward(self, A_previous):
        """
        Forward propagation step. Take the (prediction) A from the previous
        layer, and predict this layer`s A with a corresponding activation function.
        :param A_previous: Prediction of the previous layer
        :return:
        """
        self.Z = np.dot(
            self.W,
            A_previous
        ) + self.b
        self.A = self.activation(self.Z)

        return self.A

    def backward(self, samples, dA_previous, dZ_previous, A_next, W_previous=None):
        """
        Backward propagation step. Take the previous layer`s dA, dZ, W
        and calculate... whatever.
        :return:
        """
        # If the layer is last in the network (self.last state),
        # backward propagation is not needed
        if self.last:
            self.dA = dA_previous
            self.dZ = dZ_previous
        else:
            self.dA = np.dot(W_previous.T, dA_previous * dZ_previous)
            self.dZ = self.activation_derivative(self.A)

        self.dW = np.dot(self.dA * self.dZ, A_next.T) / samples
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / samples


class Input(Layer):
    """
    Input layer. Child of the Layer class.
    __init__ requires the full layer shape,
    because it is the first layer in the Network
    and the next layer depends on its shape.
    :param data: Input data (expected to be normalized and vectorized).
    :param shape: Shape of the data (tuple of ints).
    """

    def __init__(self, data, shape: tuple):
        super().__init__('Input', shape, activation=None, last=False)
        self.A = data

    def initialize(self, previous_shape):
        pass

    def get_shape(self):
        return self.shape[0]

    def get_n_of_samples(self):
        return self.shape[-1]

    def forward(self):
        """
        Forward step for the Input layer.
        Requires no operations, just pass the input data to the next layer.
        :return: data
        """
        return self.A


class Network:
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = []
        if any(layers):
            for layer in layers:
                self.add_layer(layer)

        #assert layers[0] is Input, 'First layer has to be an Input!'
        self.samples = self.layers[0].get_n_of_samples()


    def add_layer(self, layer):
        # TODO: def add_layer()
        if any(self.layers):
            # shape of the new layer will be determined
            # by the shape of the previous layer
            prev_shape = self.get_shape(-1)
            layer.initialize(prev_shape)
        # else:
        # self.layers is empty
        # this layer is the first,
        # and it is the Input layer
        # assert layer is Input

        self.layers.append(layer)

    def get_shape(self, index: int):
        """
        Return shape of the layer by its index in the layers array.
        :param index: int Index of the layer to get the shape of.
        :return: Shape of the given layer (tuple of ints)
        """
        # TODO: write scenarios for the Layer subclasses
        return self.layers[index].get_shape()

    def forward_propagation(self):
        """
        Forward propagation step. Call forward() for every layer
        in the self.layers with the result of forward()
        of the previous layer.
        :return: Prediction (A) of the last layer.
        """
        A_previous = self.layers[0].forward()
        for layer in self.layers[1:]:
            A_previous = layer.forward(A_previous)

        return self.layers[-1].A

    def backward_propagation(self, dAL, dZL):
        """
        Backward propagation step. Call backward() for every layer
        in self.layers with the result of backward()
        of the previous layer.
        """
        A_next = self.layers[-2].A
        self.layers[-1].backward(self.samples, dAL, dZL, A_next)
        for index in range(1, len(self.layers) - 1):
            dA_previous = self.layers[-index].dA
            dZ_previous = self.layers[-index].dZ
            A_next = self.layers[-2-index].A
            W_previous = self.layers[-index].W
            self.layers[-1-index].backward(self.samples, dA_previous, dZ_previous, A_next, W_previous)

    def update_parameters(self, learning_rate):
        """
        Update parameters for each layer:
            W -= learning_rate * dW
        :param learning_rate: low - slower learning, more accurate
                              large - faster learning, may be less accurate
        """
        # Start with 1 because the 1st layer is an Input
        for index in range(1, len(self.layers)):
            self.layers[index].W -= learning_rate * self.layers[index].dW
