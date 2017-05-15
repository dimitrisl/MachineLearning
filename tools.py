import numpy


def activationFunction(activationFunctionName, inputHL):
   
    if activationFunctionName == "logSoftPlus":
        result = numpy.log(1+numpy.exp(inputHL))

    elif activationFunctionName == "tanh":
        result = (numpy.exp(inputHL)-numpy.exp(-inputHL))/(numpy.exp(inputHL)+numpy.exp(-inputHL))
        
    elif activationFunctionName == "cosine":
        result = numpy.cos(inputHL)
             
    return result


def softmax(inputX):
    return numpy.exp(inputX) / numpy.sum(numpy.exp(inputX), axis=0)


def gradderivatives(actFunction, z):
    if actFunction == "logSoftPlus":
        result = (numpy.exp(z)/numpy.exp(z)+1)
    elif actFunction == 'tanh':
        result = 1 - (numpy.square(numpy.exp(z)-numpy.exp(-z))/numpy.square(numpy.exp(z)+numpy.exp(-z)))
    elif actFunction == 'cosine':
        result = -numpy.sin(z)
    return result
