import numpy
import os


train_path = os.path.join(os.getcwd(), "mnisttxt")

train = {}
test = {}
for root, directories, files in os.walk(train_path):
    for f in files:
        print f
        name = f.replace(".txt", "")
        f1 = open(train_path+os.sep+str(f))
        data = [line.split(" ") for line in f1.readlines()]
        bias = numpy.array(numpy.ones(len(data)))
        if "train" in f:
            train[name] = numpy.column_stack((bias, data))
        else:
            test[name] = numpy.column_stack((bias, data))
        f1.close()


print train["train0"][0]
print test["test0"][0]