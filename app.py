import streamlit as st
import joblib
import numpy as np

# تحميل الموديل
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button to get a prediction")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

gender = st.selectbox("Enter Gender", ["Male", "Female"])

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

predictbutton = st.button("Predict")

st.divider()

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]
    X_array = np.array([X])

    X_scaled = scaler.transform(X_array)

    prediction = model.predict(X_scaled)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()
    st.write(f"Predicted Churn: {predicted}")