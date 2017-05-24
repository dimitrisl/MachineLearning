from costfunction import costFunction

threshold = 10

user_function = raw_input("Please choose one of the activation functions: 1)logSoftPlus 2)tanh 3)cosine")
user_units = input("Choose the number of activation units from : 100, 200, 300, 400, 500")

difference = 100

while difference < threshold:
    cost, previous_instance = costFunction(user_units, user_function)
    print cost
