import numpy as np
from tools import softmax


def costFunction(W2, Z2, reg, T):
    A3 = Z2.dot(W2.T)
    Y = softmax(A3)
    E = np.log(Y).T.dot(T)-reg
    return E