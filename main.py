import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import skimage.color as color

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('colorization_model.h5')
    return model

model = load_model()

# Preprocess the uploaded grayscale image
def preprocess_image(image):
    # Resize image to 32x32 and convert to grayscale
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = image.reshape(1, 32, 32, 1)  # Reshape for model input
    return image

# Postprocess the predicted ab channels and convert to RGB
def postprocess_output(grayscale_input, ab_output):
    ab_output = ab_output * 128  # Rescale ab channels back to original range
    grayscale_input = grayscale_input * 100  # Rescale L channel to original range
    lab_image = np.concatenate((grayscale_input, ab_output), axis=-1)
    rgb_image = color.lab2rgb(lab_image[0])
    return rgb_image

st.title("Image Colorization App")

st.write("Upload a grayscale image to see it colorized by the model.")

# File uploader for grayscale image
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded grayscale image
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(img, caption='Uploaded Grayscale Image', use_column_width=True)
    
    # Preprocess the image for the model
    img_array = preprocess_image(img)
    
    # Predict the ab channels using the model
    with st.spinner('Colorizing...'):
        predicted_ab = model.predict(img_array)
    
    # Convert grayscale + predicted ab to RGB image
    colorized_image = postprocess_output(img_array[0], predicted_ab[0])
    
    # Display the colorized image
    st.image(colorized_image, caption='Colorized Image', use_column_width=True)
