
#  Heart Disease Prediction App

A Machine Learning-powered Streamlit web app that predicts the presence of heart disease based on patient data. Users can choose between a **Random Forest** model and an **Artificial Neural Network (ANN)** model for prediction.

---

##  Features

- Clean and responsive **Streamlit UI**
- Choose between two powerful ML models:
  - Random Forest (high accuracy, fast)
  -  ANN (deep learning with scaled input)
- Real-time prediction
- Model confidence and output display

---

##  Tech Stack

- **Frontend**: Streamlit
- **Backend Models**: 
  - `random_forest_model.pkl` (Scikit-learn)
  - `ann_model.h5` (TensorFlow/Keras)
- **Scaler**: `scaler.pkl` (StandardScaler from Scikit-learn)
- **Language**: Python

---

##  Getting Started (Local Setup)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-predictor.git
   cd heart-disease-predictor

2.Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate  # On Windows

3.Install dependencies:
pip install -r requirements.txt

4.Run the app:
streamlit run app.py




