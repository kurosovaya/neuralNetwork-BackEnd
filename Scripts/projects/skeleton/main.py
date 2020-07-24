import numpy
import pymongo
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask_cors import CORS
import random
from datetime import datetime, date, time
import network
from customListConverter import ListConverter


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.url_map.converters['list'] = ListConverter
CORS(app)

mongo_client = pymongo.MongoClient(port=27017)
db = mongo_client["neuralNetwork"]


#возврат предсказание по имени пользователя
@app.route("/return_by_name/<string:user_name>/")
def return_prediction_byname(user_name):
    try:
        user_data = (db.users.find_one({"user_name": user_name}))["user_data"]
        user_data = list(map(float, user_data))

        result = network.returnPrediction(user_data)
        return jsonify({"result": result})
    except:
        return jsonify({"result": None})


#вернуть предсказание по id пользователя
@app.route("/return_by_id/<int:user_id>/")
def return_prediction(user_id):
    user_data = (db.users.find_one({"_id": user_id}))["user_data"]
    user_data = list(map(float, user_data))
    result = network.returnPrediction(user_data)
    return jsonify({"result": result})


#создание пользователя
@app.route("/create_user/<string:user_name>/<list:user_data>/")
def create_user(user_name, user_data):
    user_id = 0
    sortedDB = db.users.find({}).sort("_id", -1).limit(1)
    if sortedDB.count() != 0:
        for item in sortedDB:
            user_id = (item["_id"]) + 1

    db.users.insert_one({"user_name": user_name, "user_data": user_data, "_id": user_id})
    return jsonify({"name": user_name, "data": user_data, "_id": user_id,
                    "dateOfCreation": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})


#создать товар
@app.route("/create_good/<string:goods_name>/<list:goods_tags>")
def add_goods(goods_name, goods_tags):
    try:
        goods_tags = list(map(int, goods_tags))
        network.addGoods(goods_name, goods_tags)
        print ("Yay")
    except Exception as error:
        print('Error: ' + repr(error))


#повысить вес товара
@app.route("/increase_weight/<string:user_name>/<int:goods_id>")
def increase_weight(user_name, goods_id):
    try:
        user_data = (db.users.find_one({"user_name": user_name}))["user_data"]
        user_data = list(map(float, user_data))
        network.increaseWeight(user_data, goods_id)
    except Exception as error:
        print('Error: ' + repr(error))


#понизить вес товара
@app.route("/decrease_weight/<string:user_name>/<int:goods_id>")
def decrease_weight(user_name, goods_id):
    try:
        user_data = (db.users.find_one({"user_name": user_name}))["user_data"]
        user_data = list(map(float, user_data))
        network.decreaseWeight(user_data, goods_id)
    except Exception as error:
        print('Error: ' + repr(error))


#сохранить веса в БД
def safe_widths(weights_arr):
    db.weights.insert_one({"weights_0_1": weights_arr[0], "weights_1_2":weights_arr[1],
                           "dateOfCreation": datetime.now().strftime("%Y.%m.%d %H:%M:%S")})


#safe_widths(network.learnNetwork())

if __name__ == "__main__":
    app.run()
