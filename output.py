import numpy as np
from tools import softmax
from InputLayer import inputlayer
from hidden import hiddenlayer


def sumnorm(W1, W2):
    squares1 = np.sum(np.square(W1), axis=0)
    sum1 = np.sum(squares1)
    squares2 = np.sum(np.square(W2), axis=0)
    sum2 = np.sum(squares2)
    return sum1+sum2


def costFunction(W1, W2, Y, T, reg=1):
    E = np.sum(np.sum(np.multiply(np.log(Y), T), axis=1))-(reg/2)*sumnorm(W1, W2)
    return E

X, T, W1, W2 = inputlayer()
Y = hiddenlayer()

W = sumnorm(W1, W2)
print costFunction(W1, W2, Y, T)
