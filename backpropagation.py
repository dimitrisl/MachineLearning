import numpy as np
from output import outputlayer
from tools import gradderivH


def error_output(Y, T):
     return T-Y# gradient error in output layer or layer 3


def out_grad_weights_w2(Z2, W2, d_out, reg):
    UpdW2 = Z2.T.dot(d_out)+reg.dot(W2) # Z is the output of the hidden layer and d_out the error_output
    return UpdW2

def out_grad_weights_w1(W2, d_out, W1 ,X, Z1 ,actFunction, reg):
    d_1 = np.multiply(gradderivH(actFunction, Z1), d_out.dot(W2.T))
    updW1 = X.T.dot(d_1) - reg.dot(W1)
    return updW1


def backpropagate(X, Y, T, Z1, Z2, W1, W2, d_out,actFunction):
    d_out= error_output(Y, T)
    UpdW2 = out_grad_weights_w2(Z2, W2, d_out, reg=1)
    UpdW1 = out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction, reg=1)
    return UpdW2, UpdW1