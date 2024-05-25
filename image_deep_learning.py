"""
Author: Tega Osowa
Email: stevetega.osowa11@gmail.com
GitHub: OsoTega
Description: This is a python script that uses a deep learning CNN architecture to
classify images of pets.
License: Opensource and free to use
"""

import tensorflow as tf
import numpy as np
import json
import cv2
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, models
from random import uniform, randint
# from tensorflow.python.keras import layers, models
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Function to load and preprocess the image by converting to a numpy array
def load_image(url):
    image = Image.open(url).convert('RGB')

    # Resize the loaded image
    resized_image = image.resize((224, 224))

    # Normalized pixels
    normalized_image = np.array(resized_image).astype(np.float32) / 225.0

    return normalized_image


# Function calls the load_image function, and then matches the loaded image to labels
def parse_image(filename, label):

    # process as numpy array, and return a tensor
    image = tf.numpy_function(load_image, [filename], tf.float32)
    # Provide the shape of the resulting process, suitable for the model
    image.set_shape((224, 224, 3))
    return image, label


# This Function retrieves the location of all the training dataset from the specified folder
def load_image_data(folder):
    file_paths, labels = [], []
    categories = ['Cat', 'Dog']

    for category in categories:
        directory = os.path.join(folder, category)
        label = categories.index(category)
        for filename in os.listdir(directory):
            if not filename.endswith('.db'):
                file_paths.append(os.path.join(directory, filename))
                labels.append(label)

    return file_paths, labels


# Function creates the dataset for training, given the filepath array, and the label array
def create_dataset(file_paths, labels, batch_size=32):
    # Generate a tensor with the filepath and the labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # For each data in the dataset, call the function parse_image on it,
    # reading and converting the file path to the image data, while optimizing computation
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the dataset for better learning
    dataset = dataset.shuffle(buffer_size=len(file_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Function creates the model architecture, with 3 CNN and two dense layers
def create_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')
    ])

    # Binary Crossentropy for classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function trains the model using the dataset and the architecture, and then test the model with and image
def train_dog_and_cat_classification():
    images_input, labels_input = load_image_data(os.path.join(ROOT_DIR, 'AI_Datasets/PetImages/'))
    dataset = create_dataset(images_input, labels_input)

    model = create_model()
    model.fit(dataset, epochs=50, batch_size=32)

    image = load_image("./AI_Datasets/PetImages/Dog/40.jpg")
    image = np.expand_dims(image, axis=0)
    predicted_classification = model.predict(image)
    model.save('pet_classification.h5')
    print(predicted_classification)


# Load the saved model, and predict with the model
def predict_pet_classification():
    model = tf.keras.models.load_model('pet_classification.h5')
    image = load_image("./AI_Datasets/PetImages/Dog/9.jpg")

    # expand_dims is used to add another dimension, according to the data
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
