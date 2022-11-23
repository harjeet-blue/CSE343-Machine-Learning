class Network:

    def __init__(self):
        self.epoch_loss = []
        self.layers = []
        self.loss_prime = None
        self.val_loss = []
        self.loss = None

    # set loss to use
    def loss_function(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input

    def predict(self, input_data):
        final_result = []
        # run network over all samples

        for i in range(len(input_data)):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            final_result.append(output)

        return final_result


    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # train the network
    def fit(self, x_train, y_train, x_val, y_val, epochs, learning_rate):

        # sample dimension first
        sample_size = len(x_train)

        self.epoch_loss.clear()
        self.val_loss.clear()

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples

            err /= sample_size
            self.epoch_loss.append(err)                  #calculation of training loss per epoch
            
            out = self.predict(x_val)                    # calculation validation loss per epoch
            err_val = 0
            
            for k in range(len(x_val)):
                err_val += self.loss(y_val[k], out[k])

            self.val_loss.append(err_val/len(x_val))

            print('NeuralNet:-  epoch no %d/%d   learning rate = %f   error/loss = %f   val_loss = %f' % (i+1, epochs, learning_rate, err, err_val/len(x_val)))


