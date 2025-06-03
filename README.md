
# 🍕 Pizza vs Ice Cream Classifier

A CNN-based image classifier that predicts whether a given image is **Pizza** or **Ice Cream** using **TensorFlow**, **ResNet50**, and deployed with **Gradio on Hugging Face Spaces**.

## 🚀 Live Demo  
👉 [Click here to try the app](https://osammmy-image-classification.hf.space/?__theme=system&deep_link=JdLrnZ-nMso)

## 🧠 Model Architecture
- **Base Model**: ResNet50 (ImageNet weights, frozen)
- **Custom Layers**: GlobalAveragePooling + Dense + Dropout
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam (tunable learning rate)

## 📦 Dataset
Custom dataset with image augmentation including:
- Horizontal flip  
- Rotation  
- Grayscale conversion  
- Gaussian blur  

## 🛠️ Tech Stack
- Python
- TensorFlow/Keras
- OpenCV
- Gradio
- Hugging Face Spaces

## 🔮 Prediction Output
> "This is Pizza" or "This is Ice Cream"  

## 📂 Project Structure
```
📁 dataset/
📁 augmented_dataset/
📄 best_model.keras
📄 app.py
📄 requirements.txt
```

## ✨ Author
**Samuel Oni**  
[LinkedIn](https://www.linkedin.com/in/samuel-oni) | [GitHub](https://github.com/oni-samuel)
