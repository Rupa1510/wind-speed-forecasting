import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost + Lag Features)")
st.markdown("Upload your wind data CSV file to get predictions using lag-based model.")

# === Load model ===
model = None
model_path = "xgb_wind_model_lag.pkl"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error("❌ Model file not found. Please upload 'xgb_wind_model_lag.pkl'.")

# === Upload CSV ===
uploaded_file = st.file_uploader("📤 Upload CSV with these columns: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
        st.subheader("📄 Uploaded Data:")
        st.dataframe(df.head())

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Required columns missing. Please include: {required_cols}")
        else:
            # Sort if datetime is available
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                df = df.sort_values(by='DateTime')

            # Create lag features
            df['lag_1'] = df['100m_N_Avg'].shift(1)
            df['lag_2'] = df['100m_N_Avg'].shift(2)
            df['lag_3'] = df['100m_N_Avg'].shift(3)

            # Drop NaNs created by shift
            df.dropna(inplace=True)

            # Select features
            features = ['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']
