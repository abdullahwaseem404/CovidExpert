import os

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 224

MODEL_PATH = os.path.join(
    "models",
    "best_model.h5"
)

st.set_page_config(

    page_title="CovidExpert AI",

    page_icon="🩺",

    layout="centered"
)

st.title(
    "🩺 CovidExpert AI - X-Ray Diagnosis"
)

st.write(
    "AI-based chest X-ray classification"
)

st.warning(
    "⚠️ Educational/research demo only. "
    "This application is not a medical diagnosis "
    "and should not be used for medical decisions."
)

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model not found."
    )

    st.info(
        "Please run the following command first:"
    )

    st.code(
        "python train.py",
        language="bash"
    )

    st.write(
        f"Expected model location: `{MODEL_PATH}`"
    )

    st.stop()

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()

uploaded_file = st.file_uploader(

    "Upload a chest X-ray",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file is not None:

    file_bytes = np.asarray(

        bytearray(
            uploaded_file.read()
        ),

        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:

        st.error(
            "❌ Could not read this image."
        )

        st.stop()

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    st.image(

        img_rgb,

        caption="Uploaded X-Ray",

        use_container_width=True
    )

    resized = cv2.resize(

        img_rgb,

        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    processed = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    input_image = np.expand_dims(
        processed,
        axis=0
    )

    prediction = model.predict(

        input_image,

        verbose=0
    )[0][0]

    normal_probability = float(
        prediction
    )

    covid_probability = (
        1.0 - normal_probability
    )

    st.divider()

    st.subheader(
        "Prediction"
    )


    if normal_probability >= 0.5:

        label = "Normal 😃"

        confidence = (
            normal_probability
        )

        st.success(
            f"Prediction: {label}"
        )

    else:

        label = "COVID 😷"

        confidence = (
            covid_probability
        )

        st.error(
            f"Prediction: {label}"
        )

    st.metric(

        "Confidence",

        f"{confidence * 100:.2f}%"
    )

    st.subheader(
        "Probability Breakdown"
    )

    col1, col2 = st.columns(2)


    with col1:

        st.metric(

            "COVID",

            f"{covid_probability * 100:.2f}%"
        )


    with col2:

        st.metric(

            "Normal",

            f"{normal_probability * 100:.2f}%"
        )

    st.write(
        "COVID probability"
    )

    st.progress(
        covid_probability
    )


    st.write(
        "Normal probability"
    )

    st.progress(
        normal_probability
    )