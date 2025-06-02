import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("best_model.keras")

# Define class names
class_names = ["Ice Cream", "Pizza"]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = preprocess_input(img_array)  # Make sure model was trained with ResNet50
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    label = class_names[int(np.round(prediction))]
    confidence = float(prediction if label == "Pizza" else 1 - prediction)
    return label, confidence

# Streamlit app UI
st.title("🍕 vs 🍦 Classifier")
st.write("Upload an image of either **pizza** or **ice cream**, and I’ll tell you which one it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(image)

    st.success(f"🧠 This is most likely a **{label}** with {confidence:.2%} confidence.")
