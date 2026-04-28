import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras_tuner as kt # KerasTuner is needed because build_model uses its 'hp' object

# --- Configuration Parameters (must match your training setup) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
num_classes = 4
class_names = ['Benign', '[Malignant] Pre-B', '[Malignant] Pro-B', '[Malignant] early Pre-B'] # Assuming these are your class names

# --- Define the build_model function (required for loading with custom_objects) ---
# This function must exactly match the one used during KerasTuner search
def build_model(hp):
    base_model_ht = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

    base_model_ht.trainable = True

    # KerasTuner uses an 'hp' object to define hyperparameters. When loading, we provide a dummy one.
    # The actual values for fine_tune_at, dropout_rate, num_dense_units, learning_rate will come from the loaded model.
    # These default values here are just placeholders for the 'hp' object's structure.
    fine_tune_at = hp.Int('fine_tune_at', min_value=0, max_value=200, step=50, default=0)

    for layer in base_model_ht.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model_ht(inputs, training=False) # Set training=False for inference
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.1)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    num_units = hp.Int('num_dense_units', min_value=128, max_value=512, step=128, default=256)
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model_ht = tf.keras.models.Model(inputs, outputs)

    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5], default=1e-4)

    model_ht.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['accuracy'])
    return model_ht

# Create a dummy HyperParameters object for loading the model
# This is necessary because the build_model function expects an `hp` argument.
class DummyHyperParameters(kt.HyperParameters):
    def Int(self, name, min_value, max_value, step=None, default=None):
        return default if default is not None else min_value
    def Float(self, name, min_value, max_value, step=None, default=None):
        return default if default is not None else min_value
    def Choice(self, name, values, default=None):
        return default if default is not None else values[0]

dummy_hp = DummyHyperParameters()

# --- Load the trained model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.keras',
                                          custom_objects={'build_model': build_model})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Streamlit App Interface ---
st.title("Blood Cell Cancer Classification")
st.write("Upload an image of a blood cell to classify it as Benign or Malignant.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((IMG_WIDTH, IMG_HEIGHT))) # Resize
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = tf.cast(img_array / 255.0, tf.float32) # Rescale

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"The image most likely belongs to **{predicted_class}** with a **{confidence:.2f}%** confidence.")

    st.subheader("Prediction Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, score.numpy())):
        st.write(f"{class_name}: {100 * prob:.2f}%")
