from costfunction import costFunction
from InputLayer import inputlayer
from hidden import hiddenlayer
from output import outputlayer
from backpropagation import backpropagate
import numpy
import os
import re

def load_data():
    train_path = os.path.join(os.getcwd(), "mnisttxt")
    train = {}  # Input training data
    test = {}
    labels = {}
    for root, directories, files in os.walk(train_path):  # read the input files
        for f in files:
            name = f.replace(".txt", "")
            f1 = open(train_path + os.sep + str(f))
            data = [map(int, line.split(" ")) for line in f1.readlines()]
            for lista in range(len(data)):
                data[lista] = [float(element) / 255 for element in data[lista]]
            bias = numpy.array(numpy.ones(len(data)))
            if "train" in f:
                train[name] = numpy.column_stack((bias, data))
                array = [0 for i in range(10)]
                label = int(re.search(r'\d+', f).group())
                array[label] = 1
                labels[label] = [array for i in range(len(train[name]))]
            else:
                test[name] = numpy.column_stack((bias, data))
            f1.close()
    temp = []
    X = []
    for i in range(10):
        temp.extend(labels[i])
        X.extend(train["train" + str(i)])
    return X, temp, test

X, T, Xtest = load_data()
threshold = 5


activation_function = raw_input("Please choose one of the activation functions: 1)logSoftPlus 2)tanh 3)cosine : ")
M = input("Choose the number of activation units from : 100, 200, 300, 400, 500 : ")
error = 0
error_prev = -numpy.inf

X, T, W1, W2 = inputlayer(M, X, T)
n = 0.5/X.shape[0]
iter = 1000
for epoch in range(iter):
    print "-------------epoch %s-------------" %epoch
    Z1, A2 = hiddenlayer(X, W1, activation_function)
    Y, Z2 = outputlayer(A2, W2)
    E = costFunction(Y, T, W1, W2)
    print 'Error returned: ', E
    UpW2, UpW1 = backpropagate(X, Y, T, Z1, A2,  W1, W2, activation_function)
    if numpy.absolute(error - error_prev) < threshold:
        break
    W1 += n*UpW1
    W2 += n*UpW2
    error_prev = E