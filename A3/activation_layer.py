import numpy as np
from layer import Layer
# inherit from base class Layer


class Softmax(Layer):

    def forward_propagation(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, output_error.size)
        return self.output * np.dot(output_error, np.identity(output_error.size) - out)



class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error





