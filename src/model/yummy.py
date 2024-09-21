from flask import Flask, request, jsonify, send_file
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

print("Loading model..")
model_name = 'F:\diplomski\Spas-Za-Has\models\second.3.49-2.79.keras'
with tf.device('/cpu:0'):
    model = load_model(model_name)

print("Loading metadata..")

class_to_index = {}
index_to_class = {}

with open('../../dataset/raw/Food-101/meta/classes.txt', 'r') as file:
    classes = [line.strip() for line in file.readlines()]
    class_to_index = dict(zip(classes, range(len(classes))))
    index_to_class = dict(zip(range(len(classes)), classes))
    class_to_index = {v: k for k, v in index_to_class.items()}

def model_predict(image_path, model):
    my_image = image.load_img(image_path, target_size=(299,299))
    array_image = image.img_to_array(my_image)
    array_image = np.expand_dims(array_image, axis=0)
    array_image = preprocess_input(array_image)

    predictions = model.predict(array_image)

    N = 5
    top_N_indices = predictions[0].argsort()[-N:][::-1]
    top_N_labels = []
    for i in top_N_indices:
        label = index_to_class[i].replace('_', ' ')
        confidence = predictions[0][i] * 100
        formatted_confidence = f"{confidence:.2f}%"
        top_N_labels.append((label, formatted_confidence))
    
    return top_N_labels

@app.route('/predicthard')
def predict_hardcoded():
    file = '../../dataset/raw/Food-101/images/apple_pie/0a2fe41f46838c3ecdb3309bf1395d29.jpg'

    top_N_labels = model_predict(file, model)

    return jsonify(top_N_labels)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = "./" + file.filename
        file.save(file_path)
        
        top_N_labels = model_predict(file_path, model)
        
        os.remove(file_path)
        
        return jsonify(top_N_labels)

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run()