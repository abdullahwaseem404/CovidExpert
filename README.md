# 🩺 CovidExpert AI – X-Ray Diagnosis

CovidExpert AI is a deep learning-based web application for classifying chest X-ray images as COVID or Normal.
It uses transfer learning with CNN architectures and provides confidence scores through an interactive Streamlit interface.

## 🚀 Features

📷 **Chest X-Ray Upload**

* Upload JPG, JPEG, or PNG X-ray images
* Display the uploaded X-ray before prediction

🤖 **AI-Powered Classification**

* COVID vs Normal classification
* Confidence score for predictions
* Probability breakdown for both classes

🧠 **Deep Learning Models**

* ResNet50
* EfficientNetB0
* DenseNet121

🏆 **Best Model**

* DenseNet121 achieved the highest test accuracy
* Test Accuracy: **94.74%**
* Precision: **0.95**
* Recall: **0.95**
* F1-Score: **0.95**

⚡ **Streamlit Application**

* Simple and responsive interface
* Loads the trained DenseNet model
* Displays prediction and confidence results

## 🧠 Model Training

All three CNN architectures use transfer learning with ImageNet-pretrained weights.

The models are trained for binary classification:

* `0` → COVID
* `1` → Normal

Training uses:

* Image size: 224 × 224
* Adam optimizer
* Binary cross-entropy loss
* Early stopping
* Model checkpointing

## 📊 Model Performance

| Model           | Test Accuracy |
| --------------- | ------------: |
| ResNet50        |        73.68% |
| EfficientNetB0  |        73.68% |
| **DenseNet121** |    **94.74%** |

DenseNet121 was selected as the final model based on test accuracy.

## ⚙️ Installation

```bash
git clone https://github.com/abdullahwaseem404/CovidExpert.git
pip install -r requirements.txt
```

## ▶️ Run the App

First train the models using `train.ipynb`.

The best model weights are saved to:

```text
models/best_model.weights.h5
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Dataset Structure

Organize the dataset as:

```text
dataset/
├── covid/
└── normal/
```

The current training dataset contains **94 X-ray images**:

* 69 COVID images
* 25 Normal images

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Streamlit

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic system and should not be used to make medical decisions.

---
