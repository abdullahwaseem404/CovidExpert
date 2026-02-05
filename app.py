import streamlit as st
import cv2
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import joblib


st.set_page_config(page_title="COVID-19 X-Ray Predictor", layout="wide")
st.title("🩺 COVID-19 X-Ray Predictor")
st.write("Upload an X-ray image, and the model will predict if it's COVID or Normal.")


MODEL_PATH = "mlp_covid_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found! Please train and save it as 'mlp_covid_model.pkl'.")
    st.stop()

mlp = joblib.load(MODEL_PATH)
categories = ["COVID", "Normal"]


uploaded_file = st.file_uploader("Upload X-ray image (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (150, 150))

    st.image(img_resized, caption="Uploaded X-ray", width=250)

    X_input = img_resized.reshape(1, -1)
    prediction = mlp.predict(X_input)[0]
    prob = mlp.predict_proba(X_input)[0]

    st.subheader("Prediction Result")
    st.write(f"**Class:** {categories[prediction]}")
    st.write(f"**Probability:** COVID: {prob[0]:.2f}, Normal: {prob[1]:.2f}")