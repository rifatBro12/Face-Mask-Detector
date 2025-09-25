# app.py

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load model
model = load_model("mask_detector_model.h5")

st.title("Face Mask Detector 😷")
st.write("Upload an image, and the app will predict if the person is wearing a mask or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to array
    image = np.array(image.convert("RGB"))

    # Preprocess
    image_resized = cv2.resize(image, (128, 128))
    image_scaled = image_resized / 255.0
    image_reshaped = np.expand_dims(image_scaled, axis=0)

    # Predict
    prediction = model.predict(image_reshaped)
    pred_label = np.argmax(prediction)

    # Display result
    if pred_label == 1:
        st.success("✅ The person in the image is wearing a mask 😷")
    else:
        st.error("❌ The person in the image is NOT wearing a mask")

    st.write("Raw prediction:", prediction)
else:
    st.info("Please upload an image to get a prediction.")
