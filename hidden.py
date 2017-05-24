import numpy
from InputLayer import inputlayer
from tools import activationfunction, gradderivH


def hiddenlayer(M, function_name):
    function_name = function_name# activation function
    #Calculate Z1
    X, T, W1, W2 = inputlayer(M)
    Z1 = X.dot(W1.T)
    #Call activation function
    H = activationfunction(function_name, Z1)
    A2 = numpy.c_[numpy.ones((H.shape[0], 1)), H]
    #Calculate Z2
    Z2 = A2.dot(W2.T)

    return Z1, H, Z2, (X, T, W1, W2)
