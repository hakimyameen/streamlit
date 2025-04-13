import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model2.pkl")  # Change filename if needed

# App UI
st.set_page_config(page_title="💊 Health Insurance Premium Predictor", layout="centered")
st.title("💊 Health Insurance Premium Cost Predictor")
st.markdown("🔍 Fill in the details below to estimate your insurance charges:")

# User Inputs
age = st.slider("🎂 Age", 18, 100, 30)
sex = st.radio("🧍 Sex", ['Male ♂️', 'Female ♀️'])
bmi = st.slider("⚖️ BMI (Body Mass Index)", 10.0, 50.0, 22.0)
children = st.number_input("👶 Number of Children", min_value=0, max_value=10, step=1)
smoker = st.radio("🚬 Do you smoke?", ['Yes 🚭', 'No 👍'])

# Encode inputs
sex_encoded = 1 if 'Male' in sex else 0
smoker_encoded = 1 if 'Yes' in smoker else 0

# Predict Button
if st.button("🧮 Predict My Premium"):
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Insurance Charges: **${prediction:,.2f}**")
    st.balloons()
