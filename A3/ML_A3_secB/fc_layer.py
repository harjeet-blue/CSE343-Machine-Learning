import numpy as np
from layer import Layer
# inherit from base class Layer


class FullyConnectedLayer(Layer):

    # input_size = number of input neurons
    # output_size = number of output neurons

    def __init__(self, input_size, output_size, initializer):

        if initializer == 'random':
            self.weights = np.random.rand(input_size, output_size) - 0.5                 # randome initializer
            self.bias = np.random.rand(1, output_size) - 0.5


        elif initializer == 'normal':
            self.weights = np.random.normal( 0 , 1, (input_size, output_size))           # normal with mean 0 and varience 1
            self.bias = np.random.normal( 0, 1, (1, output_size))

        else:
            self.weights = np.zeros((input_size, output_size))                           # zero init
            self.bias = np.random.rand(1, output_size)


    #updating the parameters
    def update_parameters(self, lr, wt_error, out_error):
        self.weights -= lr * wt_error
        self.bias -= lr * out_error

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.dot_product(self.input, self.weights) + self.bias
        return self.output

   
    def dot_product(self, x , y):
            return np.dot(x, y)

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.

    def backward_propagation(self, output_error, learning_rate):
        input_error = self.dot_product(output_error, self.weights.T)
        weights_error = self.dot_product(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.update_parameters(learning_rate, weights_error, output_error)

        return input_error


