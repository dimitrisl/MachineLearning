import numpy as np
from costfunction import costFunction
from InputLayer import inputlayer
from hidden import hiddenlayer
from output import outputlayer
from backpropagation import backpropagate

    
def gradCheck(activation_function,M):
    epsilon=1e-4
    X,T=load_data()
    X, T, W1, W2 = inputlayer(M,X,T)
    listSample = np.random.randint(X.shape[0], size=10)
    XSample = np.array(X[listSample, :])
    TSample = np.array(T[listSample, :])
    Z1, A2 = hiddenlayer(XSample, W1, activation_function)
    Y, Z2 = outputlayer(A2, W2)
#    E = costFunction(Y, TSample, W1, W2)
    UpW2, UpW1 = backpropagate(XSample, Y, TSample, Z1, A2,  W1, W2, activation_function)
    
    print "gradw2 : ", UpW2.shape
    print "gradw1 : ", UpW1.shape
    numGradW1 = np.zeros(UpW1.shape)
    numGradW2 = np.zeros(UpW2.shape)
    
    # gradcheck for w1
    for k in range(0, numGradW1.shape[0]):
        for d in range(0, numGradW1.shape[1]):
            tmpW = np.copy(W1)
            tmpW[k, d] += epsilon
            EPlus= costFunction(Y, TSample, tmpW, W2)

            tmpW = np.copy(W1)
            tmpW[k, d] -= epsilon
            EMinus=costFunction(Y, TSample, tmpW, W2)
            numGradW1[k, d] = (EPlus - EMinus) / (2 * epsilon)
            
    # Absolute norm
    print "The difference estimate for gradient of W1 is : ", np.amax(np.abs(UpW1 - numGradW1))
    
    # gradcheck for w2
    for k in range(0, numGradW2.shape[0]):
        for d in range(0, numGradW2.shape[1]):
            tmpW = np.copy(W2)
            tmpW[k, d] += epsilon
            EPlus= costFunction(Y, TSample, W1, tmpW)

            tmpW = np.copy(W2)
            tmpW[k, d] -= epsilon
            EMinus=costFunction(Y, TSample, W1, tmpW)
            numGradW2[k, d] = (EPlus - EMinus) / (2 * epsilon)
            
    # Absolute norm
    print "The difference estimate for gradient of W2 is : ", np.amax(np.abs(UpW2 - numGradW2))

gradCheck("tanh",100)