import numpy
import os
import re
import random
train_path = os.path.join(os.getcwd(), "mnisttxt")

train = {}
test = {}
labels = []
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
            labels.extend([array for i in range(len(train[name]))])
        else:
            test[name] = numpy.column_stack((bias, data))
        f1.close()

y_labels = numpy.array(labels)
print y_labels
M = random.choice([100, 200, 300, 400, 500])
weights1 = numpy.random.randn(len(train["train0"][0]), M)
