import numpy as np
import pymongo
import os
from datetime import datetime

np.random.seed(228)

mongo_client = pymongo.MongoClient(port=27017)
db = mongo_client["neuralNetwork"]

alpha = 0.05

def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


def sigmoid(x):
    return 1 / (1 + np.exp2(-x))


def tanh(x):
    return np.tanh(x)
    

def tanh2deriv(output):
    return 1 - (output ** 2)


def learnNetwork():
    hidden_size = 256
    users_data = []
    output_check = []

    #f = open(os.getcwd() + "/Scripts/projects/skeleton/DeleteThatPls/usersDataSet.txt")
    f = open("usersDataSet.txt")
    raw_clients = f.readlines()
    f.close()

    #f = open(os.getcwd() + "/Scripts/projects/skeleton/DeleteThatPls/GoodsDataSet.txt")
    f = open("GoodsDataSet.txt")
    raw_goods = f.readlines()
    f.close()

    for items in raw_clients:
        splited_data = list(map(float, items.split(" ")))
        users_data.append(splited_data)

    goods_names = raw_goods.pop(0).split(" ")
    for i in range(len(goods_names)):
        db.goods.insert_one({"name": goods_names[i], "tag_id": [int(i/11)], "_id": i})

    for items in raw_goods:
        splited_data = list(map(float, items.split(" ")))
        output_check.append(splited_data)


    input_lenght = len(users_data[0])
    output_lenght = len(output_check[0])

    users_data = np.array(users_data)
    output_check = np.array(output_check)

    weights_0_1 = 2 * np.random.rand(input_lenght, hidden_size) - 1
    weights_1_2 = 2 * np.random.rand(hidden_size, output_lenght) - 1

    for iteration in range(3):
        for i in range(len(users_data)):
            layer_0 = np.array(users_data[i : i + 1])            
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

            layer_2_error = np.sum((layer_2 - output_check[i : i + 1]) ** 2)


            answer = output_check[i:i+1]

            for j in range (len(layer_2)):            
                layer_2_delta = layer_2[j] - answer[j]
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)

                layer2_error = alpha * layer_1.T.dot(layer_2_delta[j])
                layer1_error = alpha * layer_0.T.dot(layer_1_delta)

                weights_1_2 -= layer2_error 
                weights_0_1 -= layer1_error
            if i % 100 == 1:
                 #pass
                print("Error:" + str(layer_2_error))
                #print(answer)
                #print(layer_2)
                     

    totalErr = []
    print("Final test")
    for i in range(len(users_data)):        
        layer_0 = np.array(users_data[i : i + 1])
        layer_1 = tanh(np.dot(layer_0, weights_0_1))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_error = np.sum((layer_2 - output_check[i]) ** 2)
        totalErr.append((layer_2_error))
        if i % 100 == 1:
            print("Error:" + str(layer_2_error))
    print(np.sum(totalErr))
    safe_widths([weights_0_1.tolist(), weights_1_2.tolist()])


def returnPrediction(user_data):
    goods_data = db.goods.find({})

    weights_arr = db.weights.find({}).sort("dateOfCreation", -1).limit(1)
    weights_0_1 = None
    weights_1_2 = None
    for item in weights_arr:
        weights_0_1 = item["weights_0_1"]
        weights_1_2 = item["weights_1_2"]

    layer_0 = np.array(user_data)
    layer_1 = tanh(np.dot(layer_0, weights_0_1))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
    output = [[layer_2[i], goods_data[i]["name"], False, goods_data[i]["_id"]] for i in range(len(layer_2))]
    user_tags = user_data[2:]
    for i in range(len(output)):
        if user_tags[goods_data[i]["tag_id"][0]] >= 0.5:
            output[i][2] = True
    output.sort(reverse = True)
    return (output[0:16])

def addGoods(goods_name, goods_tags):
    id = db.goods.find({}).sort("_id", -1).limit(1)[0]["_id"] + 1
    db.goods.insert_one({"name": goods_name, "tag_id": goods_tags, "_id": id})

    weights_arr = db.weights.find({}).sort("dateOfCreation", -1).limit(1)
    weights_0_1 = None
    weights_1_2 = None

    for item in weights_arr:
        weights_0_1 = item["weights_0_1"]
        weights_1_2 = item["weights_1_2"]
    safe_widths([weights_0_1, np.hstack((weights_1_2,
                np.random.rand(1,len(weights_1_2)).T)).tolist()])

def safe_widths(weights_arr):
    db.weights.insert_one({"weights_0_1": weights_arr[0], "weights_1_2":weights_arr[1],
                           "dateOfCreation": datetime.now().strftime("%Y.%m.%d %H:%M:%S")})


def increaseWeight(user_data, goods_id):
    weights_arr = db.weights.find({}).sort("dateOfCreation", -1).limit(1)
    weights_0_1 = None
    weights_1_2 = None
    for item in weights_arr:
        weights_0_1 = np.array(item["weights_0_1"])
        weights_1_2 = np.array(item["weights_1_2"])

    pers_weight_1_2 = weights_1_2.T[goods_id]

    for i in range(1):
        layer_1 = tanh(np.dot(user_data, weights_0_1))
        layer_2 = sigmoid(np.dot(layer_1, pers_weight_1_2))

        layer_2_delta = layer_2 - 1
        print(layer_2_delta)
        layer_1_delta = np.dot(layer_2_delta, pers_weight_1_2) * tanh2deriv(layer_1)

        layer_2_error = layer_2_delta * 0.5
        layer_1_error = layer_1_delta * 0.001

        pers_weight_1_2 -= layer_2_error
        weights_0_1 -= layer_1_error

    weights_1_2.T[goods_id : goods_id + 1] = pers_weight_1_2
    safe_widths([weights_0_1.tolist(), weights_1_2.tolist()])


def decreaseWeight(user_data, goods_id):
    weights_arr = db.weights.find({}).sort("dateOfCreation", -1).limit(1)
    weights_0_1 = None
    weights_1_2 = None
    for item in weights_arr:
        weights_0_1 = np.array(item["weights_0_1"])
        weights_1_2 = np.array(item["weights_1_2"])

    pers_weight_1_2 = weights_1_2.T[goods_id]

    for i in range(1):
        layer_1 = tanh(np.dot(user_data, weights_0_1))
        layer_2 = sigmoid(np.dot(layer_1, pers_weight_1_2))

        layer_2_delta = layer_2 - 0
        print(layer_2_delta)
        layer_1_delta = np.dot(layer_2_delta, pers_weight_1_2) * tanh2deriv(layer_1)

        layer_2_error = layer_2_delta * 0.5
        layer_1_error = layer_1_delta * 0.01

        pers_weight_1_2 -= layer_2_error
        weights_0_1 -= layer_1_error

    weights_1_2.T[goods_id : goods_id + 1] = pers_weight_1_2
    safe_widths([weights_0_1.tolist(), weights_1_2.tolist()])


#decreaseWeight([0.1,1,0,0,0,0,1,0,0,0,0], 100)
#learnNetwork()

#db.weights.remove( { "dateOfCreation": { "$gt": "2020.06.07 18:46:31" } })