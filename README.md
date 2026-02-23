# Forest-Fire-Detector
Engineered a low-latency CNN-based wildfire detection system using MobileNetV2 transfer learning for binary satellite image classification. Built a custom data standardization pipeline to resolve TensorFlow encoding issues. Achieved 97.2% validation accuracy with strong generalization and fast, deployment-ready inference.

AI-Based Wildfire Detection using MobileNetV2 (Transfer Learning).
# Overview

This project presents a low-latency, high-accuracy automated system for binary wildfire detection in satellite imagery. The model leverages Transfer Learning with MobileNetV2 (ImageNet pretrained) to classify images as Wildfire or No Wildfire.

A custom data standardization pipeline was engineered to resolve TensorFlow JPEG encoding inconsistencies, ensuring stable and efficient model training. The final model achieved 97.2% validation accuracy with strong generalization and fast inference speed, making it suitable for deployment in resource-constrained environments.

# Key Features

Binary classification: Wildfire vs. No Wildfire

Transfer Learning using MobileNetV2

Custom image preprocessing & JPEG standardization pipeline

Optimized for low latency inference

Deployment-ready lightweight architecture

# Tech Stack

Python

TensorFlow / Keras

MobileNetV2 (ImageNet Pretrained)

NumPy

OpenCV

Matplotlib

# Dataset

Satellite wildfire imagery dataset

Binary labeled images (Wildfire / No Wildfire)

Stratified Train–Validation split

Preprocessed and standardized before training

# Model Architecture

Base Model: MobileNetV2 (ImageNet pretrained)

Frozen base layers + fine-tuned top layers

Custom dense classification head

Loss Function: Binary Cross-Entropy

Optimizer: Adam

# Data Preprocessing Pipeline

Image resizing to model input dimensions

Pixel normalization

Non-standard JPEG format correction

Dataset validation and encoding consistency checks

# Performance

Validation Accuracy: 97.2%

Strong generalization across diverse wildfire scenarios

Compact model size

Fast inference suitable for real-time applications



# Training
python train.py

# Inference
python predict.py --image sample.jpg


Output:

Prediction: Wildfire (Confidence: 0.98)

# Deployment Potential

Suitable for edge devices

Compatible with cloud-based disaster monitoring systems

Can be integrated into real-time satellite image pipelines

# Future Improvements

Multi-class fire severity detection

Real-time satellite feed integration

Ensemble modeling for further accuracy improvement

Explainable AI (Grad-CAM visualization)

# Installation & Setup
1️⃣ Clone the Repository
git clone https: https://github.com/AyuskaSaha/Forest-Fire-Detector

`cd your-repo-name`

2️⃣ Create a Virtual Environment (Recommended)

`python -m venv venv`


Activate it:

Windows

`venv\Scripts\activate`


Mac/Linux

`source venv/bin/activate`

3️⃣ Install Dependencies
`pip install -r requirements.txt`

▶️ Running the Project
🔹 Train the Model
`python train.py`

🔹 Run Inference on an Image
`python predict.py` --image sample.jpg


Expected Output:

Prediction: Wildfire (Confidence: 0.97)
