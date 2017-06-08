import numpy as np
from tools import gradderivH


def error_output(Y, T):
    # delta error on output layer
    delta_output = T-Y
    return delta_output


def out_grad_weights_w2(W2, d_out, A2, reg=1):

    # Update rule for W2 weights
    UpdW2 = np.subtract(d_out.T.dot(A2), reg*W2)
    print 'Dimension of updated W2 array is ', UpdW2.shape

    return UpdW2


def out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction, reg=1):
    # Calculate derivative of activation function
    grad_actfunction = gradderivH(actFunction, Z1)
    # Add bias
    grad_actfunction = np.c_[np.ones((grad_actfunction.shape[0], 1)), grad_actfunction]

    # delta error on hidden layer
    d_1 = np.multiply(d_out.dot(W2), grad_actfunction)
    print 'Dimension of delta error hidden is ', d_1.shape, type(d_1)

    # Update rule for W1 weights
    UpdW1 = np.subtract(d_1.T.dot(X), reg*W1)
    print 'Dimension of updated W2 array is ', UpdW1.shape
    return UpdW1


def backpropagate(X, Y, T, Z1, A2,  W1, W2, actFunction):
    d_out = error_output(Y, T)
    UpdW2 = out_grad_weights_w2(W2, d_out, A2)
    UpdW1 = out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction)
    return UpdW2, UpdW1

