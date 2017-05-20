import numpy as np
from output import outputlayer
from tools import gradderivH


def error_output(Y, T):
    delta_out = T-Y# gradient error in output layer or layer 3
    return delta_out


def out_grad_weights(Z2, W2, d_out, reg):

    UpdW2 = Z2.T.dot(d_out)+reg*W2# Z is the output of the hidden layer and d_out the error_output
    return UpdW2


def error_hidden(W2, d_out, Z1, W1, reg):
    #
    pass
