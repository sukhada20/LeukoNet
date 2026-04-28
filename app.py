import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "Benign",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B",
    "[Malignant] early Pre-B"
]

MODEL_PATH = "best_efficientnet_model.keras"   # or .h5 if you saved that

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(image):
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image)

    # Ensure 3 channels (RGB)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0   # SAME as training
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# UI
# -----------------------------
st.title("🧬 Blood Cancer (ALL) Classification")
st.write("Upload a microscopic blood cell image to classify leukemia type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction:")
    st.success(f"{predicted_class}")

    st.subheader("Confidence:")
    st.write(f"{confidence * 100:.2f}%")

    # Show all probabilities
    st.subheader("Class Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")
