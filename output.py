import numpy as np
from hidden import hiddenlayer


def softmax(inputX):
    m = np.ndarray.max(inputX, axis=1)
    numstab = inputX-m[:, None]
    x_exp = np.exp(numstab)
    denominator = np.ndarray.sum(x_exp, axis=1)
    return x_exp/denominator[:, None]


def outputlayer(M, function_name):
    Z1, H, Z2, input_properties, A2 = hiddenlayer(M, function_name)
    Y = softmax(Z2)
    return Y, Z1, Z2, input_properties, A2
