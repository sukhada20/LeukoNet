import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 🔥 IMPORTANT: Register EfficientNet before loading model
from tensorflow.keras.applications import EfficientNetB0

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

MODEL_PATH = "clean_model.keras"   # ✅ use re-saved model

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False   # 🔥 fixes your error
    )
    return model

model = load_model()

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image)

    image = image / 255.0  # SAME as training
    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Leukemia Classifier", layout="centered")

st.title("🧬 Blood Cancer (ALL) Classification")
st.write("Upload a microscopic blood cell image to classify leukemia type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(processed_image)

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction:")
    
    if "Malignant" in predicted_class:
        st.error(f"⚠️ {predicted_class}")
    else:
        st.success(f"✅ {predicted_class}")

    st.subheader("Confidence:")
    st.write(f"{confidence * 100:.2f}%")

    # Show probabilities
    st.subheader("Class Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only and not for clinical diagnosis.")
