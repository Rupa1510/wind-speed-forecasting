import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# App setup
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost + Lag Features)")
st.markdown("Upload your wind data CSV file below to get predictions using lag-based XGBoost model.")

# Try loading model
model = None
model_path = "xgb_wind_model_lag.pkl"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully using joblib.")
    except:
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            st.success("✅ Model loaded successfully using XGBoost native loader.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ Model file not found. Please upload 'xgb_wind_model_lag.pkl' in the same folder.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV with columns: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')  # or encoding='ISO-8859-1'

        st.subheader("📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Required columns missing. Please include: {required_cols}")
        else:
            # Sort if datetime available
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df = df.sort_values(by='DateTime')

            # Add lag features
            df['lag_1'] = df['100m_N_Avg'].shift(1)
            df['lag_2'] = df['100m_N_Avg'].shift(2)
            df['lag_3'] = df['100m_N_Avg'].shift(3)

            # Drop rows with NaNs due to lag
            df.dropna(inplace=True)

            # Select features
            features = ['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']
            X_new = df[features]

            # Prediction
            if model is not None:
                df['Predicted_100m_N_Avg'] = model.predict(X_new)

                # Show output
                st.subheader("📊 Prediction Results:")
                if 'DateTime' in df.columns:
                    st.dataframe(df[['DateTime'] + features + ['Predicted_100m_N_Avg']])
                else:
                    st.dataframe(df[features + ['Predicted_100m_N_Avg']])

                # Download
                csv_output = df.to_csv(index=False)
                st.download_button("📥 Download Predictions as CSV", csv_output, file_name="wind_speed_predictions.csv", mime="text/csv")
            else:
                st.warning("⚠️ Model is not loaded. Cannot predict.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("ℹ️ Upload a valid CSV file to begin prediction.")

st.markdown("---")
st.caption("Developed by Rupa • XGBoost Forecasting with Lag Features ⚡")
