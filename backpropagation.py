import numpy as np
from output import outputlayer
from tools import gradderivH


def error_output(Y, T):
    delta_out = T-Y# gradient error in output layer or layer 3
    return delta_out


def out_grad_weights_w2(Z2, W2, d_out, reg):

    UpdW2 = Z2.T.dot(d_out)+reg*W2# Z is the output of the hidden layer and d_out the error_output
    return UpdW2

def out_grad_weights_w1(Z2, W2, d_out, reg, W1 ,X, z1 ,actFunction):
    UpdW1=Z2.T.dot(W2).dot(gradderivH(actFunction, z1)).dot(X)-reg*W1
    return UpdW1
    
def error_hidden(W2, d_out, Z1, W1, reg):
    #
    pass
