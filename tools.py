import numpy

def activationFunction(activationFunctionName,inputHL):
   
    if activationFunctionName=="logSoftPlus":
        result=numpy.log(1+numpy.exp(inputHL))
        
    
    elif activationFunctionName=="tanh":
        result=(numpy.exp(inputHL)-numpy.exp(-inputHL))/(numpy.exp(inputHL)+numpy.exp(-inputHL))
        
    elif activationFunctionName=="cosine":
        result=numpy.cos(inputHL)
             
    return result