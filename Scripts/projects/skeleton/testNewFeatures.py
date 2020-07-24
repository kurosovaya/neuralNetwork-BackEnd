#test
import numpy as np
import os

np.random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp2(-x))


def sigmoid2deriv(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)
    

alpha = 0.05

hidden_size = 64
users_data = []
output_check = []

f = open(os.getcwd() + "/Scripts/projects/skeleton/usersDataSet.txt")
#f = open("usersDataSet.txt")
raw_clients = f.readlines()
f.close()

f = open(os.getcwd() + "/Scripts/projects/skeleton/GoodsDataSet.txt")
#f = open("GoodsDataSet.txt")
raw_goods = f.readlines()
f.close()

for items in raw_clients:
    splited_data = list(map(float, items.split(" ")))
    users_data.append(splited_data)

goods_names = raw_goods.pop(0).split(" ")

for items in raw_goods:
    splited_data = list(map(float, items.split(" ")))
    output_check.append(splited_data)

input_lenght = len(users_data[0])
output_lenght = len(output_check[0])

users_data = np.array(users_data)
output_check = np.array(output_check)

rng_state = np.random.get_state()
np.random.shuffle(users_data)
np.random.set_state(rng_state)
np.random.shuffle(output_check)


weights_0_1 = 0.02 * np.random.rand(input_lenght, hidden_size) - 0.01
weights_1_2 = 0.2 * np.random.rand(hidden_size, output_lenght) - 0.1
batch_size = 64

for iteration in range(10):
    for i in range(int(len(users_data) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

        layer_0 = np.array(users_data[batch_start : batch_end])        
        layer_1 = tanh(np.dot(layer_0, weights_0_1))

        dropout_mask  = np.random.randint(2, size = layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_error = np.sum((output_check[batch_start : batch_end] - layer_2) ** 2)

        answer = output_check[batch_start : batch_end]
        
        layer_2_delta = (answer - layer_2) * sigmoid2deriv(layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)            

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        if i % 100 == 1:
            pass
            # pass
            #print("Error:" + str(layer_2_error))
            # print(answer)
            # print(layer_2)

totalErr = []
print("Final test")
for i in range(len(users_data)):
    layer_0 = np.array(users_data[i : i + 1])
    layer_1 = tanh(np.dot(layer_0, weights_0_1))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    layer_2_error = np.sum((output_check[i] - [layer_2]) ** 2)
    totalErr.append((layer_2_error))
    if i % 100 == 1:
        print("Error:" + str(layer_2_error))
        #print("prediction: " + str(layer_2))
        #print("answer: " + str(output_check[i]))
print(np.sum(totalErr))
