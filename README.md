
# Potato Disease Classification Using CNN 🌿🥔
This project focuses on classifying potato leaf diseases using a Convolutional Neural Network (CNN). The goal is to develop a deep learning model that can accurately identify diseases from images of potato plant leaves. The model is built using Keras and TensorFlow.

# Project Overview
Potato plants are prone to several diseases that can significantly impact yield. Early detection of these diseases can help farmers take timely actions and prevent major crop loss. This project aims to use deep learning to classify potato leaf diseases, specifically targeting diseases like:
      
      1) Late Blight
      2) Early Blight
      3) Healthy Leaves

# Dataset
The dataset used in this project is sourced from Kaggle's PlantVillage Dataset.

# Model Architecture
The model is built using a Convolutional Neural Network (CNN), which is commonly used for image classification tasks. The architecture consists of:

- Multiple Convolutional Layers for feature extraction
- MaxPooling Layers for down-sampling
- Dense Layers for classification
- ReLU as activation function and Softmax for output

# Installation

To get started with this project, follow these steps:

### Clone the repository

`git clone https://github.com/Rajshree1102/Potato_Disease_Detection`

### Install dependencies

Make sure you have Python 3.7+ installed. Install the required packages:

`pip install -r requirements.txt`

# Running the Web Application
A Streamlit web application has been developed for potato disease classification. You can upload an image of a potato leaf, and the model will predict if the leaf is healthy or diseased.

### To run the app:
`streamlit run potato_disease_web.py`

# Results
- Accuracy on Training Set: 97.7%

- Accuracy on Test Set: 98.2%

      






