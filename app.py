import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# CONFIG
# =========================
IMG_SIZE = 224  # change if your training used a different size

CLASS_NAMES = [
    "Benign",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B",
    "[Malignant] early Pre-B"
]

# =========================
# LOAD MODEL (FIXED)
# =========================
@st.cache_resource
def load_model():
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model = tf.keras.models.load_model(
        "best_efficientnet_model.h5",   # or .h5 if needed
        custom_objects={
            "EfficientNetB0": EfficientNetB0,
            "preprocess_input": preprocess_input
        },
        compile=False  # 🔥 avoids deserialization errors
    )
    return model

model = load_model()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(image):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    image = preprocess_input(image)  # 🔥 correct for EfficientNet

    image = np.expand_dims(image, axis=0)
    return image

# =========================
# UI
# =========================
st.set_page_config(page_title="Leukemia Detection", layout="centered")

st.title("🧬 Leukemia Detection using EfficientNet")
st.write("Upload a blood smear image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(processed_image)

    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions))

    # =========================
    # RESULTS
    # =========================
    st.success(f"🧾 Prediction: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2%}**")

    # =========================
    # PROBABILITY BREAKDOWN
    # =========================
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob:.4f}")
