import numpy as np

# loss function and its derivative
# cross-entropy loss

def cross_entropy(y_true, y_pred):
    return np.dot( np.log(y_pred), -1*y_true)


def cross_entropy_derivative(y_true, y_pred):
    return (-1*y_true)/y_pred


# 0-1 loss

def L2_loss(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def L2_loss_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


