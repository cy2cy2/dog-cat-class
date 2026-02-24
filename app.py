import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Set page config
st.set_page_config(page_title="Cat vs Dog Image Classifier", page_icon="🐾")

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or a dog, and the model will predict which one it is!")

@st.cache_resource
def load_model():
    # Load the trained model
    model = tf.keras.models.load_model('best_model_xception.keras')
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    desired_size = (model.input_shape[1], model.input_shape[2]) if model.input_shape[1] is not None else (299, 299)
    img_resized = img.resize(desired_size)
    
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    # Depending on how the model was trained, it might need Rescaling or specific preprocessing. 
    # Usually, Xception expects values between -1 and 1 via tf.keras.applications.xception.preprocess_input
    # or it might have a Rescaling layer inside the model. 
    # Let's assume the model handles its own scaling or expects standard 0-255 images (if Rescaling layer is present) 
    # or needs xception preprocessing.
    # We will try passing the raw array. If it fails or is wildly inaccurate, we may need to adjust preprocessing.
    
    try:
        predictions = model.predict(img_array)
        
        # Assuming binary classification where 0=Cat, 1=Dog, OR a 2-class softmax.
        # Let's interpret the prediction
        if predictions.shape[-1] == 1:
            # Binary classification (e.g., sigmoid activation)
            score = predictions[0][0]
            if score > 0.5:
                # If dog is 1 and cat is 0
                st.success(f"It's a **Dog**! (Confidence: {score:.2f})")
            else:
                st.success(f"It's a **Cat**! (Confidence: {1 - score:.2f})")
        else:
            # Categorical classification (e.g., softmax activation)
            cat_score = predictions[0][0] # Assuming 0 is Cat
            dog_score = predictions[0][1] # Assuming 1 is Dog
            if cat_score > dog_score:
                st.success(f"It's a **Cat**! (Confidence: {cat_score:.2f})")
            else:
                st.success(f"It's a **Dog**! (Confidence: {dog_score:.2f})")
                
    except Exception as e:
        st.error(f"Error during prediction: {e}")
