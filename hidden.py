import numpy
from tools import activationfunction


def hiddenlayer(X, W1, function_name):
    function_name = function_name  # activation function

    # Calculate Z1 the array containing the input to the hidden unit
    Z1 = X.dot(W1.T)
    print 'Dimension of array Z1: ', Z1.shape

    # Call activation function
    H = activationfunction(function_name, Z1)
    print 'Dimension of array H after applying activation function remains: ', H.shape

    # Add bias unit on hidden layer
    A2 = numpy.c_[numpy.ones((H.shape[0], 1)), H]
    print 'Dimension of array A2: ', A2.shape

    return Z1, A2
