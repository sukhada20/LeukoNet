import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.h5"
IMG_SIZE = 224

CLASS_NAMES = ["ALL", "HEM"]  # change if your classes differ

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Leukonet", page_icon="🧬", layout="centered")

st.title("🧬 Leukonet - Blood Cancer Detection")
st.write("Upload a microscopic blood cell image to classify it.")

uploaded_file = st.file_uploader(
    "📤 Upload an Image", 
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed)[0]

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("🧾 Result")

    if predicted_class == "ALL":
        st.error(f"⚠️ Prediction: {predicted_class}")
    else:
        st.success(f"✅ Prediction: {predicted_class}")

    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # -----------------------------
    # PROBABILITY BAR
    # -----------------------------
    st.subheader("📊 Class Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {predictions[i]*100:.2f}%")
        st.progress(float(predictions[i]))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚠️ This is an AI-based tool and not a medical diagnosis.")
