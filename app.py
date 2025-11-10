import streamlit as st
import pickle
import pandas as pd

# Load the saved model
model_path = r"C:\Users\Hrucha\AdCampaignProject\models\my_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("📊 My Machine Learning Model Deployment")

# Example: let user upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Data Uploaded Successfully")
    st.dataframe(data.head())

    # Run prediction
    predictions = model.predict(data)
    st.write("### 🔮 Predictions")
    st.write(predictions)