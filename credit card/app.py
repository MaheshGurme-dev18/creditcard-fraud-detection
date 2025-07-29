import streamlit as st
import pickle
import numpy as np

# Load model
with open("creditcard.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction data below. The model will predict if it's **fraudulent (1)** or **normal (0)**.")

# Input fields
time = st.number_input("Time (seconds)", value=10000)
amount = st.number_input("Amount ($)", value=100.00)

# Input for V1 to V28 features
v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0, format="%.5f")
    v_features.append(v)

# Create final input vector
input_data = [time] + v_features + [amount]
input_array = np.array([input_data])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")

