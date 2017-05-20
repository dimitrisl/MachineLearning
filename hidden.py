import numpy
from InputLayer import inputlayer
from tools import activationfunction, gradderivH

X, T, W1, W2 = inputlayer()
def hiddenlayer():
    #Calculate Z1
    Z1 = X.dot(W1.T)

    #Call activation function
    H = activationfunction("logSoftPlus", Z1)
    A2 = numpy.c_[numpy.ones((H.shape[0], 1)), H]
    #Calculate Z2
    Z2 = A2.dot(W2.T)

    return Z1, H, Z2

a, b, c = hiddenlayer()
grad = gradderivH("logSoftPlus", c)
print grad