import numpy as np
from tools import softmax



def outputlayer(A2, W2):

    # Calculate Z2 which is the array containing the input of the output layer
    Z2 = A2.dot(W2.T)
    print 'Dimension of array Z2: ', Z2.shape

    # Calculate Y which is the output of the output unit after activating softmax on Z2
    Y = softmax(Z2)
    print 'Dimension of array Y: ', Y.shape

    return Y, Z2
