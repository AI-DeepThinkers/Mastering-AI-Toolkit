import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🧠 Handwritten Digit Classifier")
st.write("Upload a 28x28 grayscale image of a digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image).resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(image, caption="Processed Input", width=150)
    st.write(f"🔢 **Predicted Digit:** {predicted_class}")
    st.bar_chart(prediction[0])
