import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# App Title
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("🧭 Enter passenger details below and click **Predict** to see if they'd survive the Titanic disaster!")

# Sidebar input section
st.sidebar.header("🔍 Input Features")

pclass = st.sidebar.selectbox('🎫 Passenger Class', [1, 2, 3], format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else f"{x}rd Class"))
sex = st.sidebar.radio('👤 Sex', ['male', 'female'])
age = st.sidebar.slider('🎂 Age', 0, 100, 30)
sibsp = st.sidebar.slider('🧍‍🤝‍🧍 Siblings/Spouses Aboard', 0, 10, 0)
fare = st.sidebar.slider('💰 Fare', 0.0, 500.0, 50.0)

# Convert categorical to numeric
sex_encoded = 0 if sex == 'male' else 1

# Prepare input
input_data = pd.DataFrame({
    'pclass': [pclass],
    'sex': [sex_encoded],
    'age': [age],
    'sibsp': [sibsp],
    'fare': [fare]
})

# Display the input summary
with st.expander("📋 Passenger Summary"):
    st.write(f"**Class:** {pclass}")
    st.write(f"**Sex:** {sex}")
    st.write(f"**Age:** {age}")
    st.write(f"**Siblings/Spouses:** {sibsp}")
    st.write(f"**Fare:** ${fare:.2f}")

# Predict Button
if st.button('🚀 Predict Survival'):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"🎉 The passenger **survived**! (🟢 {proba[1]*100:.2f}% confidence)")
        st.balloons()
    else:
        st.error(f"💀 Unfortunately, the passenger **did not survive**. (🔴 {proba[0]*100:.2f}% confidence)")

st.markdown("---")
st.markdown("📘 *Built with love using Streamlit*")
