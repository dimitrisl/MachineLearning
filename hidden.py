import numpy
from InputLayer import inputlayer
from tools import activationfunction, gradderivH


def hiddenlayer(M, function_name):
    function_name = function_name# activation function
    #Calculate Z1
    X, T, W1, W2 = inputlayer(M)
    print W1.shape
    print X.shape
    Z1 = X.dot(W1.T)
    print Z1.shape
    #Call activation function
    H = activationfunction(function_name, Z1)
    print "to shape einai ",H.shape
    A2 = numpy.c_[numpy.ones((H.shape[0], 1)), H]
    print A2.shape
    print W2.shape
    #Calculate Z2
    Z2 = A2.dot(W2.T)
    print Z2.shape
    return Z1, H, Z2, (X, T, W1, W2)
