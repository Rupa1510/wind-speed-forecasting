import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# App setup
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost + Lag Features)")
st.markdown("Upload your wind data CSV file below to get predictions.")

# Load model
model_path = "xgb_wind_model_lag.pkl"  # ensure this is your lag-trained model
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please upload 'xgb_wind_model_lag.pkl'.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV with: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data:")
        st.dataframe(df.head(10))

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Your CSV must contain these columns: {required_cols}")
        else:
            # Optional sorting
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df = df.sort_values(by='DateTime')

            # Add lag features
            df['lag_1'] = df['100m_N_Avg'].shift(1)
            df['lag_2'] = df['100m_N_Avg'].shift(2)
            df['lag_3'] = df['100m_N_Avg'].shift(3)

            # Drop missing values after lag
            df.dropna(inplace=True)

            # Prepare input
            features = ['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']
            X_new = df[features]

            # Predict
            df['Predicted_100m_N_Avg'] = model.predict(X_new)

            st.subheader("📊 Prediction Results:")
            st.dataframe(df[['DateTime'] + features + ['Predicted_100m_N_Avg']] if 'DateTime' in df.columns else df)

            # Download predictions
            csv_output = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_output,
                file_name="wind_speed_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("ℹ️ Please upload a valid CSV file to get started.")

st.markdown("---")
st.caption("Developed by Rupa • Forecasting using XGBoost with Lag Features 🌐")
