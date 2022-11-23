import numpy as np
from network import Network
from fc_layer import FullyConnectedLayer
from losses import cross_entropy, cross_entropy_derivative, L2_loss, L2_loss_derivative
from activation_layer import ActivationLayer, Softmax
from activations import tanh, tanh_derivative, relu, relu_derivative, sigmoid, sigmoid_derivative, leaky_relu, leaky_relu_derivative, linear, linear_derivative


class NeuralNetwork():

    def __init__(self, layers, acitvation, initializer, loss  ,max_iter, alpha, batch_size):

        self.net = Network()                                 # objecto of network class
        self.n = len(layers)
        self.epochs = max_iter
        self.learning_rate = alpha
        self.act_fuc = acitvation
        self.loss_func = loss

        for i in range(self.n - 2):
            self.net.add_layer(FullyConnectedLayer(layers[i], layers[i+1], initializer))                   # Add a fully connected layer
            self.addActivationfun()                                                            # Add a activation layer

        self.net.add_layer(FullyConnectedLayer(layers[-2], layers[-1], initializer))
        self.net.add_layer(Softmax())                                                          # Add softmax at the output layer


    def addActivationfun(self):                          # adding the appropriate activation fucntion

        if self.act_fuc == 'tanh':
            self.net.add_layer(ActivationLayer(tanh, tanh_derivative))  

        elif self.act_fuc == 'sigmoid':
            self.net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))  

        elif self.act_fuc == 'relu':
            self.net.add_layer(ActivationLayer(relu, relu_derivative))  

        elif self.act_fuc == 'leaky_relu':
            self.net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))  

        else: 
            self.net.add_layer(ActivationLayer(linear, linear_derivative))  


    def fit(self , x_train, y_train , x_val, y_val):
        self.use_loss()
        self.net.fit(x_train, y_train, x_val= x_val, y_val=   y_val , epochs = self.epochs, learning_rate = self.learning_rate)


    def use_loss(self):         # use the loss function

        if self.loss_func == 'L2':
            self.net.loss_function(L2_loss, L2_loss_derivative)

        else:
            self.net.loss_function(cross_entropy, cross_entropy_derivative)



    def predict_prob(self, x_train):      # predict  the probability
        out = self.net.predict(x_train)
        return out
    

    def predict(self, x_test):           # predict the output
        out = self.net.predict(x_test)
        y_pred = []

        for ele in out:
            y_pred.append(np.argmax(ele))
        return y_pred


    def score(self, x, y):               # give the accurayc of the model 
        y_pred = self.predict(x)
        score = 0
        for i in range(len(x)):
            if y_pred[i] == np.argmax(y[i]):
                score += 1
        return score/len(x)
     
    def get_training_loss(self):         # to get the training loss 
        return self.net.epoch_loss;

    def get_validation_loss(self):       # to get the vaidation loss
        return self.net.val_loss;
            
