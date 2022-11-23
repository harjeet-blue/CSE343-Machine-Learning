import numpy as np
from layer import Layer

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

