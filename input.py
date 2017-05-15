import numpy
import os
import re
import random
train_path = os.path.join(os.getcwd(), "mnisttxt")


def getweights(nm, M):  # Random initialization of weights
    weights1 = 0.1 * numpy.random.randn(len(train[nm][0]), M)
    weights2 = 0.1 * numpy.random.randn(M+1, 10)
    return weights1, weights2

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
M = random.choice([100, 200, 300, 400, 500])  # Number of units in the hidden layer

W1, W2 = getweights("train0", M)  # Arrays of Weights in layer 1 and 2

