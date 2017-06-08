import numpy
import os
import re
from sklearn import preprocessing
import random


def getweights(train, M, K=10):  # Random initialization of weights
    weights1 = 0.1 * numpy.random.randn(M, len(train["train0"][0]))
    weights2 = 0.1 * numpy.random.randn(K, M+1)
    return weights1, weights2

def inputlayer(M):
    # M is the number of units in the hidden layer
    train_path = os.path.join(os.getcwd(), "mnisttxt")
    train = {}  # Input training data
    test = {}
    labels = {}
    for root, directories, files in os.walk(train_path):  # read the input files
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
        X.extend(train["train"+str(i)])

    # X is the array containing the input extended with the bias.
    X = numpy.array(X)
    T = numpy.array(temp)  # True labels of Training Examples
    W1, W2 = getweights(train, M)  # Arrays of Weights in layer 1 and 2

    print 'Dimension of arrays X and T are: ', X.shape, ' ', T.shape
    print 'Dimension of arrays W1 and W2 are: ', W1.shape, ' ', W2.shape

    # Scale all the data to values from 0-1
    # (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X = preprocessing.minmax_scale(X, feature_range=(0, 1))
    # X_scaled = X_std * (max - min) + min
    return X, T, W1, W2
