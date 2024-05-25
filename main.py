import tensorflow as tf
import numpy as np
import sys
import json
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
from random import uniform, randint
# from tensorflow.python.keras import layers, models
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def load_classify_model():
    model = load_model('pet_classification.h5')
    return model


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})


@app.route('/api/classify', methods=['POST'])
def classify():
    sys.stdout.flush()
    data = request.get_json()['data']
    prediction = None
    if data:
        float_data = np.array(data[0], dtype=np.float32)
        image = np.expand_dims(float_data, axis=0)
        model = load_classify_model()
        prediction = model.predict(image).tolist()

    return jsonify({'message': prediction})


if __name__ == '__main__':
    app.run()
