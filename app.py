import streamlit as st
import pandas as pd
import joblib
import pickle

st.title("Bank Marketing Prediction")

# Load model and preprocessing objects
model = joblib.load("rf_bank_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.write("Upload a CSV file with the same columns as training data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=';')  # use ';' if your CSV uses semicolons

    st.write("Uploaded Data:")
    st.dataframe(data.head())

    # One-hot encode categorical columns
    data_encoded = pd.get_dummies(data)

    # Add missing columns with 0
    for col in model_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Ensure columns order matches training data
    data_encoded = data_encoded[model_columns]

    # Scale numeric features
    numeric_cols = data_encoded.select_dtypes(include=['int64', 'float64']).columns
    data_encoded[numeric_cols] = scaler.transform(data_encoded[numeric_cols])

    # Make predictions
    predictions = model.predict(data_encoded)
    predictions_proba = model.predict_proba(data_encoded)[:,1]

    # Show results
    data['Prediction'] = predictions
    data['Probability'] = predictions_proba
    st.write("Predictions:")
    st.dataframe(data)
