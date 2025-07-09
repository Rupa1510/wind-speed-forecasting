import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Title
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost Model)")
st.markdown("Upload new input data and get forecasted wind speed using the trained XGBoost model.")

# Load the model
model_path = "xgb_wind_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please make sure 'xgb_wind_model.pkl' is in the same folder.")

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV file with required features (Pressure, Temp_5m, 20m_Avg)", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')


        # Check for required columns
        required_features = ['Pressure', 'Temp_5m', '20m_Avg']
        if all(feature in df_input.columns for feature in required_features):
            X_new = df_input[required_features]
            
            # Predict
            predictions = model.predict(X_new)
            df_input['Predicted_100m_N_Avg'] = predictions

            st.subheader("📊 Predictions:")
            st.dataframe(df_input)

            # Download option
            csv_output = df_input.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_output,
                file_name='wind_speed_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error(f"❌ Input file must contain these columns: {required_features}")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
else:
    st.info("ℹ️ Please upload a CSV file to proceed.")

st.markdown("---")
st.caption("Developed by Rupa for NIWE Forecasting Project 🌐")
