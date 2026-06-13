# 🩺 CovidExpert AI - X-Ray Diagnosis

CovidExpert AI is a deep learning-based web application that detects COVID-19 from chest X-ray images using state-of-the-art CNN architectures like ResNet, EfficientNet, and DenseNet.

---

## 🚀 Features

* 📷 Upload chest X-ray images
* 🤖 AI-powered COVID-19 detection
* 📊 Confidence score for predictions
* 🔥 Grad-CAM visualization for model interpretability
* ⚡ Built with Streamlit for fast deployment

---

## 🧠 Models Used

* ResNet50
* EfficientNetB0
* DenseNet121

All models use transfer learning with ImageNet weights and are fine-tuned for binary classification (COVID vs Normal).

---

## 🛠️ Installation

```bash
git clone https://github.com/abdullahwaseem404/CovidExpert.git
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🏋️ Training Models

```bash
python train.py
```

This will train ResNet, EfficientNet, and DenseNet models and save the best versions based on validation accuracy.

---

## 📊 Dataset

Organize dataset as:

```
dataset/
├── covid/
└── normal/
```

---
