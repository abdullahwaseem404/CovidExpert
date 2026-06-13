import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

st.title("🩺 CovidExpert AI - X-Ray Diagnosis")

model = tf.keras.models.load_model("resnet.h5") 

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = "COVID 😷" if pred > 0.5 else "Normal 😃"

    st.image(img[0], caption="Uploaded X-ray")
    st.write("Prediction:", label)
    st.write("Confidence:", float(pred))