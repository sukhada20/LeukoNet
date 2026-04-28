import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_efficientnet_model.keras")
    return model

model = load_model()

# =========================
# Class Names (EDIT THIS)
# =========================
CLASS_NAMES = [
    "Benign",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B",
    "[Malignant] early Pre-B"
]

IMG_SIZE = 224  # change if you used different input size

# =========================
# Preprocessing Function
# =========================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # Normalize (important!)
    image = image / 255.0

    image = np.expand_dims(image, axis=0)
    return image

# =========================
# UI
# =========================
st.title("🧠 Image Classification App")
st.write("Upload an image to predict its class")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(f"Predicted Class: **{CLASS_NAMES[predicted_class]}**")
    st.info(f"Confidence: {confidence:.2f}")

    # Show all probabilities
    st.subheader("Class Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob:.4f}")
