import streamlit as st
import numpy as np  
from predict import predict
model_path = "models/cnn_model_100.h5"
import PIL.Image as Image
import cv2

# Title and layout
st.title("Image Prediction App")
st.write("Upload a photo and get a prediction!")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))  # Load image using PIL

    # Prediction button
    if st.button("Predict"):
        prediction = predict(model_path, image)  
        st.write("Prediction:", prediction)
