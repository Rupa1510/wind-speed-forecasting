
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("xgb_wind_model.pkl")

# Streamlit UI
st.title("Wind Speed Forecasting with XGBoost")

# Input form
pressure = st.number_input("Pressure [mbar]", 900, 1100, step=1)
temp = st.number_input("Temperature at 5m [°C]", -20.0, 50.0, step=0.1)
avg_20m = st.number_input("20m Avg Wind Speed [m/s]", 0.0, 50.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[pressure, temp, avg_20m]], columns=["Pressure", "Temp_5m", "20m_Avg"])
    prediction = model.predict(input_data)
    st.success(f"Predicted 100m_N_Avg Wind Speed: {round(prediction[0], 2)} m/s")
