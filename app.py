import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model_fixed.h5"
IMG_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False   # avoids optimizer issues
    )
    return model

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# UI
# -----------------------------
st.title("🧬 LeukoNet - Leukemia Detection")

uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)

    prediction = model.predict(processed)[0][0]

    st.subheader("Prediction Result:")

    if prediction > 0.5:
        st.error(f"⚠️ Leukemia Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ Normal (Confidence: {1 - prediction:.2f})")
