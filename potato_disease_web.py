# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from PIL import Image
from io import BytesIO

#loading the saved model
loaded_model = tf.keras.models.load_model(
    'C:/Users/khush/OneDrive/Desktop/OneDrive/Desktop/self_projects/potato_disease_classification/training/train_model.keras',
    compile=False
)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy')

# creating a function for prediction



def main():
    def pred(uploaded_file):
        image = Image.open(uploaded_file)
    
    # Step 2: Convert the image to RGB (in case it's in another mode like RGBA)
        image = image.convert('RGB')
    
    # Step 3: Resize the image to the required input size (assuming 224x224 for this example)
        image = image.resize((224, 224))
    
    # Step 4: Convert the image to a NumPy array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = loaded_model.predict(img_array)
        d_class = np.argmax(predictions[0])
        if d_class == 0:
            predicted_class = 'Potato___Early_blight'
        elif d_class == 1:
            predicted_class = 'Potato___Late_blight'
        else:
            predicted_class = 'Potato___healthy'
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence
    
    #title
    st.title('Potato Disease Classification')
    
    #getting input
    st.title("Upload Image here")
    
    # Upload image using streamlit file_uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
    if uploaded_file is not None:
    # Read the uploaded file content
       file_content = uploaded_file.read()
    
    # Open the image using PIL
       image = Image.open(BytesIO(file_content))
    
    # Convert the image to a NumPy array
       image_array = np.array(image)
    
    # Display the uploaded image
       st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Process the image as needed
    # For example, display the shape of the image array
       st.write(f"Image shape: {image_array.shape}")
    else:
       st.write("Please upload an image file.")
    
    #code for prediction
    diagnosis = ''
    
    #creating a button
    if st.button('Result'):
        diagnosis = pred(uploaded_file)
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    