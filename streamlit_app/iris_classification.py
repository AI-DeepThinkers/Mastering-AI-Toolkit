# streamlit_app/iris_classification.py

import streamlit as st
import numpy as np
import joblib
import os

def iris_app():
    st.subheader("🌸 Iris Flower Species Prediction")

    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
    model = joblib.load(model_path)

    # Make prediction
    X_new = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(X_new)

    st.success(f"Predicted Species: **{prediction[0]}**")
