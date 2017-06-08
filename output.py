import numpy as np


def softmax(inputX):
    m = np.ndarray.max(inputX, axis=1)
    numstab = inputX-m[:, None]
    x_exp = np.exp(numstab)
    denominator = np.ndarray.sum(x_exp, axis=1)
    return x_exp/denominator[:, None]


def outputlayer(A2, W2):

    # Calculate Z2 which is the array containing the input of the output layer
    Z2 = A2.dot(W2.T)
    print 'Dimension of array Z2: ', Z2.shape

    # Calculate Y which is the output of the output unit after activating softmax on Z2
    Y = softmax(Z2)
    print 'Dimension of array Y: ', Y.shape

    return Y, Z2
