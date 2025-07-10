import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page title and layout
st.set_page_config(page_title="Wind Speed Forecasting", layout="wide")
st.title("🌬️ Wind Speed Forecasting App (XGBoost + Lag Features)")
st.markdown("Upload your CSV with the required features to get wind speed predictions at 100m.")

# Load model
model_path = "xgb_wind_model_lag.pkl"  # Make sure this matches your lag-trained model filename
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please ensure 'xgb_wind_model_lag.pkl' is in the same folder.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV file with: Pressure, Temp_5m, 20m_Avg, 100m_N_Avg", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)

        required_cols = ['Pressure', 'Temp_5m', '20m_Avg', '100m_N_Avg']
        if all(col in df_input.columns for col in required_cols):

            # Sort by DateTime (optional but recommended if you have time-series)
            if 'DateTime' in df_input.columns:
                df_input['DateTime'] = pd.to_datetime(df_input['DateTime'])
                df_input = df_input.sort_values(by='DateTime')

            # Create lag features from '100m_N_Avg'
            df_input['lag_1'] = df_input['100m_N_Avg'].shift(1)
            df_input['lag_2'] = df_input['100m_N_Avg'].shift(2)
            df_input['lag_3'] = df_input['100m_N_Avg'].shift(3)

            # Drop rows with NaNs from lag shifts
            df_input.dropna(subset=['lag_1', 'lag_2', 'lag_3'], inplace=True)

            # Prepare input for prediction
            X_new = df_input[['Pressure', 'Temp_5m', '20m_Avg', 'lag_1', 'lag_2', 'lag_3']]
            predictions = model.predict(X_new)

            df_input['Predicted_100m_N_Avg'] = predictions

            st.subheader("📊 Prediction Results:")
            st.dataframe(df_input)

            # Download button
            csv_output = df_input.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_output,
                file_name='wind_speed_predictions.csv',
                mime='text/csv'
            )

        else:
            st.error(f"❌ Your file must contain these columns: {required_cols}")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("ℹ️ Upload a CSV file to begin.")

# Footer
st.markdown("---")
st.caption("Developed by Rupa 🌐 | Forecasting with Lag-enhanced XGBoost Model")
