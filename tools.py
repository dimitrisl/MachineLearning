import numpy


def activationfunction(activationFunctionName, inputHL):
    if activationFunctionName == "logSoftPlus":
        result = numpy.log(1 + numpy.exp(inputHL))

    elif activationFunctionName == "tanh":
        # result = numpy.divide(numpy.exp(inputHL)-numpy.exp(-inputHL), numpy.exp(inputHL)+numpy.exp(-inputHL))
        result = (2 / (1 + numpy.exp(-2 * inputHL))) - 1  # we use tansig same results as tanh
        # result = (1-numpy.exp(numpy.multiply(-2,inputHL)))/(1+numpy.exp(numpy.multiply(-2,inputHL)))
    elif activationFunctionName == "cosine":
        result = numpy.cos(inputHL)

    return result


def softmax(inputX):
    m = numpy.ndarray.max(inputX, axis=1)
    numstab = inputX - m[:, None]  # a vector that numerically stabilized.
    x_exp = numpy.exp(numstab)
    denominator = numpy.ndarray.sum(x_exp, axis=1)
    return x_exp / denominator[:, None]


def gradderivH(actFunction, z):
    if actFunction == "logSoftPlus":
        result = (numpy.exp(z) / (numpy.exp(z) + 1))
        print result
    elif actFunction == 'tanh':
        # result = - (2*numpy.exp(z)/numpy.square(numpy.exp(z)+1))
        result = 1 - numpy.square(activationfunction('tanh', z))
    elif actFunction == 'cosine':
        result = -numpy.sin(z)
    return result