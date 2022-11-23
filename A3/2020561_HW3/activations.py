import numpy as np
# activation function and its derivative

# sigmoid 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


#relu
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.array(x > 0).astype('int')


#leaky_relu
def leaky_relu(x):
    return np.maximum(x, 0)

def leaky_relu_derivative(x):
    v = np.array(x > 0).astype('float')

    for i in range(len(v[0])):
        if v[0][i] <= 0:
            v[0][i] = 0.01
    return v

#tanh
def tanh(x):
    return np.tanh(x);

def tanh_derivative(x):
    return 1-np.tanh(x)**2;

    
#identity
def linear(x):
    return x

def linear_derivative(x):
    return np.where( x> -99999999 , 1, 0)


