import math

def activationFunction(activationFunctionName,inputHL):
   
    if activationFunctionName=="logSoftPlus":
        result=math.log(1+math.exp(inputHL))
    
    elif activationFunctionName=="tanh":
        result=(math.exp(inputHL)-math.exp(-inputHL))/(math.exp(inputHL)+math.exp(-inputHL))
        
    elif activationFunctionName=="cosine":
        result=math.cos(inputHL)
             
    return result