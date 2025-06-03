import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your trained model
model = tf.keras.models.load_model("best_model.keras")

# Define class names
class_names = ["Ice Cream", "Pizza"]

# Prediction function
def predict_image(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    label = class_names[int(np.round(prediction))]
    return label

# Gradio Interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Pizza vs Ice Cream Classifier",
    description="Upload an image of either pizza or ice cream, and the model will predict which it is!"
)

if __name__ == "__main__":
    demo.launch()
