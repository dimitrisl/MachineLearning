import numpy as np
from InputLayer import inputlayer
from hidden import hiddenlayer


def softmax(inputX):
    m = np.ndarray.max(inputX, axis=1)
    numstab = inputX-m[:, None]
    x_exp = np.exp(numstab)
    denominator = np.ndarray.sum(x_exp, axis=1)
    return x_exp/denominator[:, None]


def outputlayer():
    Z1, H, Z2 = hiddenlayer()
    Y = softmax(Z2)
    return Y
