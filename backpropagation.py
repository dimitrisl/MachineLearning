import numpy as np
from output import outputlayer
from tools import gradderivH


def error_output(Y, T):
     return T-Y# gradient error in output layer or layer 3


def out_grad_weights_w2(Z2, W2, d_out,A2, reg = 1):
    print "dout",d_out.shape
    print "A2",A2.shape
    print "w2 ",W2.shape
    print "dout",d_out.T.dot(A2).shape
    print "reg", (reg*W2).shape
    UpdW2 = np.add(d_out.T.dot(A2),reg*W2) # Z is the output of the hidden layer and d_out the error_output
    return UpdW2

def out_grad_weights_w1(W2, d_out, W1 ,X, Z1 ,actFunction, reg = 1):
    d_1 = np.multiply(gradderivH(actFunction, Z1), d_out.dot(W2))
    updW1 = np.subtract(X.T.dot(d_1), reg.dot(W1))
    return updW1


def backpropagate(X, Y, T, Z1, Z2, W1, W2, actFunction, A2):
    d_out = error_output(Y, T)
    UpdW2 = out_grad_weights_w2(Z2, W2, d_out,A2)
    UpdW1 = out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction)
    return UpdW2, UpdW1