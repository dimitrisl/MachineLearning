import numpy as np
from output import outputlayer
from InputLayer import inputlayer


def sumnorm(W1, W2):
    squares1 = np.sum(np.square(W1), axis=0)
    sum1 = np.sum(squares1)
    squares2 = np.sum(np.square(W2), axis=0)
    sum2 = np.sum(squares2)
    return sum1+sum2


def costFunction(M, function_name, reg=1):
    Y, input_properties = outputlayer(M, function_name)
    _, T, W1, W2 = input_properties #unpack
    E = np.sum(np.sum(np.multiply(np.log(Y), T), axis=1))-(reg/2)*sumnorm(W1, W2)
    return E, Y
