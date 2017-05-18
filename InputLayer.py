import numpy
import os
import re
import random


def getweights(train, nm, M, K=10):  # Random initialization of weights
    weights1 = 0.1 * numpy.random.randn(M, len(train[nm][0]))
    weights2 = 0.1 * numpy.random.randn(K, M+1)
    return weights1, weights2

def inputlayer():
    train_path = os.path.join(os.getcwd(), "mnisttxt")
    train = {}  # Input training data
    test = {}
    labels = {}
    for root, directories, files in os.walk(train_path):
        for f in files:
            name = f.replace(".txt", "")
            f1 = open(train_path+os.sep+str(f))
            data = [map(int, line.split(" ")) for line in f1.readlines()]
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
    for i in train.keys():
        X.extend(train[i])
    #X is the true input.
    X = numpy.array(X)
    T = numpy.array(temp)  # True labels of Training Examples
    #M = random.choice([100, 200, 300, 400, 500])  # Number of units in the hidden layer
    M = 500
    W1, W2 = getweights(train,"train0", M)  # Arrays of Weights in layer 1 and 2

    return X, T, W1, W2

