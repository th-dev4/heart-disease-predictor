import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load models and scaler
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

ann_model = load_model('ann_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main { background-color: #f8f8f8; padding: 2rem; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #ff5858, #f857a6);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Predictor")
st.markdown("### Enter patient health info and select the model to predict heart disease.")

# Sidebar
model_choice = st.sidebar.radio("🧠 Select a Model", ("Random Forest", "Artificial Neural Network"))

st.markdown("<div class='main'>", unsafe_allow_html=True)

# Inputs
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting ECG (restecg)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved (thalach)', min_value=60, max_value=250)
exang = st.radio('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression (oldpeak)', step=0.1)
slope = st.selectbox('Slope of ST Segment', [0, 1, 2])
ca = st.selectbox('No. of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2])  # 0=Normal, 1=Fixed, 2=Reversible

# Process gender
sex_val = 1 if sex == 'Male' else 0

# Create input array
input_data = np.array([[
    age, sex_val, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]])

# Predict button
if st.button("💡 Predict"):
    with st.spinner("Analyzing..."):
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)[0]
            accuracy = 0.9340  # Set manually or dynamically later
        else:
            scaled_input = scaler.transform(input_data)
            prediction = ann_model.predict(scaled_input)[0][0]
            prediction = int(prediction > 0.5)
            accuracy = 0.8810  # Set manually or dynamically later

    # Display result
    if prediction == 1:
        st.error("💔 **Prediction:** The person **has** heart disease.")
    else:
        st.success("💚 **Prediction:** The person **does not have** heart disease.")

    st.info(f"✅ **Model Used:** {model_choice}")
    st.code(f"🎯 Accuracy: {accuracy * 100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)
