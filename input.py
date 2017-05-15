import numpy
import os
import re
import random
train_path = os.path.join(os.getcwd(), "mnisttxt")


def getweights(nm, M):  # Random initialization of weights
    weights1 = 0.1 * numpy.random.randn(len(X[nm][0]), M)
    weights2 = 0.1 * numpy.random.randn(M+1, 10)
    return weights1, weights2

X = {}  # Input training data
test = {}
labels = {}
for root, directories, files in os.walk(train_path):
    for f in files:
        name = f.replace(".txt", "")
        f1 = open(train_path+os.sep+str(f))
        data = [map(int, line.split(" ")) for line in f1.readlines()]
        bias = numpy.array(numpy.ones(len(data)))
        if "train" in f:
            X[name] = numpy.column_stack((bias, data))
            array = [0 for i in range(10)]
            label = int(re.search(r'\d+', f).group())
            array[label] = 1
            labels[label] = [array for i in range(len(X[name]))]
        else:
            test[name] = numpy.column_stack((bias, data))
        f1.close()
temp = []
for i in range(10):
    temp.extend(labels[i])
T = numpy.array(temp)  # True labels of Training Examples
M = random.choice([100, 200, 300, 400, 500])  # Number of units in the hidden layer

W1, W2 = getweights("train0", M)  # Arrays of Weights in layer 1 and 2

