# streamlit_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the trained CNN model
#@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

# Sidebar widgets
st.sidebar.title("Handwritten Digit Recognition")
st.sidebar.write("Use this app to recognize handwritten digits (0-9) using a CNN model.")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Input widget for user interaction
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Main container
st.title("Handwritten Digit Recognition App")
st.write("Upload an image of a handwritten digit, and the app will predict the digit using a CNN model.")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    with st.spinner('Processing...'):
        processed_image = preprocess_image(image)
        time.sleep(2)  # Simulate processing time

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display progress and status updates
    st.write("Prediction in progress...")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    if confidence >= confidence_threshold:
        st.success(f"Predicted digit: {predicted_digit} with confidence: {confidence:.2f}")
    else:
        st.warning(f"Prediction confidence ({confidence:.2f}) below the threshold of {confidence_threshold}")

    # Graph to visualize the prediction probabilities
    st.write("Prediction Probabilities:")
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0], color='blue')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
else:
    st.info("Please upload an image to get started.")

# Footer
st.write("---")
