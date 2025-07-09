import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# App title and setup
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost + Lag Features)")
st.markdown("Upload new data to get predicted wind speed at 100m using lag-enhanced XGBoost model.")

# Load the model
model_path = "xgb_wind_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please place 'xgb_wind_model.pkl' in this folder.")

# Upload user input CSV
uploaded_file = st.file_uploader("📤 Upload CSV file with columns: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"❌ Required columns missing. Please include: {required_cols}")
        else:
            # ✅ Create lag features
            df_input['lag_1'] = df_input['100m_N_Avg'].shift(1)
            df_input['lag_2'] = df_input['100m_N_Avg'].shift(2)
            df_input['lag_3'] = df_input['100m_N_Avg'].shift(3)

            df_input.dropna(inplace=True)

            st.subheader("📌 Lag Feature Sample")
            st.dataframe(df_input[['lag_1', 'lag_2', 'lag_3']].head())

            # Prepare input
            X_new = df_input[['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']]
            predictions = model.predict(X_new)
            df_input['Predicted_100m_N_Avg'] = predictions

            st.subheader("📊 Forecasted Wind Speeds")
            st.dataframe(df_input[['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3', 'Predicted_100m_N_Avg']])

            csv_output = df_input.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV with Predictions",
                data=csv_output,
                file_name='predicted_wind_speed.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"⚠️ Error reading or processing file: {e}")
