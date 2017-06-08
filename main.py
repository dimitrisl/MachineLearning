from costfunction import costFunction
from InputLayer import inputlayer
from hidden import hiddenlayer
from output import outputlayer
from backpropagation import backpropagate

threshold = 0

activation_function = raw_input("Please choose one of the activation functions: 1)logSoftPlus 2)tanh 3)cosine : ")
M = input("Choose the number of activation units from : 100, 200, 300, 400, 500 : ")

epoch = 1
print "-------------epoch %s-------------" %epoch
X, T, W1, W2 = inputlayer(M)
Z1, A2 = hiddenlayer(X, W1, activation_function)
Y, Z2 = outputlayer(A2, W2)
E = costFunction(Y, T, W1, W2)
UpW2, UpW1 = backpropagate(X, Y, T, Z1, A2,  W1, W2, activation_function)
difference = E

while difference!=threshold: 
    epoch+=1
    print "-------------epoch %s-------------" %epoch
    Z1, A2 = hiddenlayer(X, UpW1, activation_function)
    Y, Z2 = outputlayer(A2, UpW2)
    E = costFunction(Y, T, UpW1, UpW2)
    UpW2, UpW1 = backpropagate(X, Y, T, Z1, A2,  UpW1, UpW2, activation_function)
    difference = difference - E
    
print E