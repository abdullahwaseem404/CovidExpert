# 🩺 CovidExpert
# COVID-19 X-Ray Classification App

A machine learning–based system for detecting **COVID-19 from chest X-ray images** using image processing and a **Neural Network (MLP)**, deployed with an interactive **Streamlit web app**.

Working Demo : https://youtu.be/D7nj08EJD7U

## 🚀 Features
- Binary classification: **COVID vs Normal**
- Image preprocessing with OpenCV (grayscale, resizing)
- Neural Network model using Scikit-learn (MLPClassifier)
- Model training, validation, and testing pipeline
- Confusion matrix and classification report
- Trained model persistence using Joblib
- Streamlit-based image upload & prediction interface

## 🧠 Model & Approach
- Images resized to **150×150** and flattened
- Train/Validation/Test split (70/15/15)
- Baseline and tuned MLP models evaluated
- Final tuned model achieved:
  - **93% Test Accuracy**
  - High recall for COVID class

## 📊 Model Performance
- **Validation Accuracy:** 100%
- **Test Accuracy:** 93.3%

## 🛠️ Tech Stack
- Python
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## ▶️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/abdullahwaseem404/CovidExpert.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

## ⚠️ Disclaimer
This project is for **educational purposes only** and should not be used for real-world medical diagnosis.
