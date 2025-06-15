# streamlit_app/mnist_cnn.py

import streamlit as st
import tensorflow as tf
# from PIL import Image, ImageOps
import numpy as np
import os

def mnist_app():
    st.subheader("🧠 Handwritten Digit Classifier (MNIST)")
    # Load test images
    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Load pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn_model.h5")
    model = tf.keras.models.load_model(model_path)

    # Select first 5 test samples
    st.markdown("### Sample Predictions from MNIST Test Set")
    predictions = model.predict(X_test[:5])

    for i in range(5):
        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(X_test[i].reshape(28, 28), width=80, clamp=True)
        with col2:
            st.markdown(f"**Predicted:** {pred_label} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; **Actual:** {true_label}")
            st.progress(float(predictions[i][pred_label]))
