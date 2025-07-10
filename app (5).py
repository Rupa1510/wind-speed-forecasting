import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page setup
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting with XGBoost + Lag Features")
st.markdown("Upload a CSV file with required features to get wind speed forecasts at 100m height.")

# Load model
model_path = "xgb_wind_model_lag.pkl"  # use your model with lag features
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please make sure 'xgb_wind_model_lag.pkl' is in the same directory.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your input CSV (with columns: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Required columns missing. Please include: {required_cols}")
        else:
            # Optional: Sort by time
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df = df.sort_values(by='DateTime')

            # Add lag features
            df['lag_1'] = df['100m_N_Avg'].shift(1)
            df['lag_2'] = df['100m_N_Avg'].shift(2)
            df['lag_3'] = df['100m_N_Avg'].shift(3)

            df.dropna(inplace=True)

            # Prepare input for prediction
            features = ['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']
            X_new = df[features]
            predictions = model.predict(X_new)

            df['Predicted_100m_N_Avg'] = predictions

            st.subheader("📊 Prediction Results:")
            st.dataframe(df)

            # Download button
            csv_output = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_output,
                file_name="wind_speed_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error reading or processing file: {e}")
else:
    st.info("ℹ️ Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Developed by Rupa • Forecasting App using XGBoost + Lag Features 🌐")
