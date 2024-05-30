import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging level to 'ERROR'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Further suppress TensorFlow logs
tf.autograph.set_verbosity(0)  # Disable verbosity from TensorFlow's AutoGraph

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import argparse

# Command-Line Arguments for Image Path
parser = argparse.ArgumentParser(description='Image Classification with Outlier Detection')
parser.add_argument('img_path', type=str, help='Path to the image file')
args = parser.parse_args()
tf.keras.backend.clear_session()

# Load Models and Define Class Labels
model = tf.keras.models.load_model('mobilenet_model.h5')
autoencoder = load_model('autoencoder.h5')
class_labels = {
    0: 'cleaning',
    1: 'gardening',
    2: 'handyman',
    3: 'pet'
}

# Function to Calculate Reconstruction Error
def calculate_reconstruction_error(autoencoder, img):
    reconstructed_img = autoencoder.predict(np.expand_dims(img, axis=0))
    error = np.mean(np.abs(reconstructed_img - img))
    return error

# Function to Determine Outlier
def is_outlier(img, autoencoder, threshold):
    error = calculate_reconstruction_error(autoencoder, img)
    return error, error > threshold

# Function to Predict Class with Outlier Detection
def predict_with_outlier_detection(model, autoencoder, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0

    error, outlier = is_outlier(img_array, autoencoder, threshold=0.15)

    if outlier:
        return "others", error, None
    else:
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_labels.get(predicted_class_idx, "unknown")
        confidence_rate = np.max(prediction)
        return predicted_class, error, confidence_rate

# Example Usage: Get Image Path from Command Line
img_path = args.img_path

# Predict and Print Results
result, error, confidence = predict_with_outlier_detection(model, autoencoder, img_path)

if confidence is not None and confidence >= 0.60:
    print(f'Predicted class: {result}')
    print(f'Confidence rate: {confidence * 100:.2f}%')
else:
    print(f'Predicted class: others')
print(f'Calculated error: {error:.4f}')
