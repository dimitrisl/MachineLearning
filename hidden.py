import numpy
from InputLayer import inputLayer


X,T,W1,W2=inputLayer()
#Calculate Z1
Z1=numpy.multiply(X,numpy.transpose(W1))

print Z1