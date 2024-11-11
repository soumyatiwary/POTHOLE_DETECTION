import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Load the CNN model
cnn_model = tf.keras.models.load_model('pothole_detector_model.h5')

# Function to preprocess the uploaded image for CNN model
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Resize to target size (224x224)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (0-1)
    return img_array

# Streamlit App UI
st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="centered")
st.title("Pothole Detection App 🚧")
st.markdown("### Upload images or take a picture to detect potholes.")
st.write(
    "This app classifies whether an image contains a pothole or is a normal road. You can upload multiple images or use your camera to take a picture."
)

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

# Camera input (optional - can only take one picture at a time)
camera_input = st.camera_input("Or take a picture")

# Collect images (either from uploaded files or camera input)
images_to_process = []

if uploaded_files:
    images_to_process.extend([Image.open(file) for file in uploaded_files])

if camera_input:
    img = Image.open(camera_input)
    images_to_process.append(img)

# Layout: Display the uploaded images side by side
if images_to_process:
    # Create columns for displaying images and results
    cols = st.columns(len(images_to_process))
    
    for idx, img in enumerate(images_to_process):
        with cols[idx]:
            # Display the image
            st.image(img, caption=f"Image {idx+1}", use_column_width=True)
            
            # Show loading spinner while processing
            with st.spinner("Processing..."):
                # Preprocess the image for CNN model prediction
                processed_img = preprocess_image(img)
                cnn_prediction = cnn_model.predict(processed_img)
                cnn_confidence = cnn_prediction[0][0] * 100  # Confidence for 'pothole' class
                
                # Determine result based on prediction threshold
                if cnn_prediction > 0.5:
                    result = f"**Pothole detected** with **{cnn_confidence:.2f}% confidence**"
                else:
                    result = f"**No pothole detected** with **{100 - cnn_confidence:.2f}% confidence**"
                
                # Display the result in a more structured format
                st.markdown(f"### Result {idx+1}")
                st.markdown(f"{result}")
                
                # Optional: Show a progress bar if processing multiple images
                st.progress((idx + 1) / len(images_to_process))

else:
    st.write("Please upload or take a picture to get started.")

