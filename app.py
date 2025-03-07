import streamlit as st
import pandas as pd
import joblib

# Load the trained model
MODEL_FILE = "biodegradable_plastic_waste_model.joblib"

def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        return None

# Function to predict biodegradable plastic waste
def predict_waste(population, plastic_consumption_rate, biodegradable_prevalence, biodegradation_rate):
    model = load_model()

    if model is None:
        return "Model file not found. Please train the model first."

    new_data = pd.DataFrame({
        "Population": [population],
        "Plastic_Consumption_Rate": [plastic_consumption_rate],
        "Biodegradable_Prevalence": [biodegradable_prevalence],
        "Biodegradation_Rate": [biodegradation_rate]
    })

    prediction = model.predict(new_data)
    return prediction[0]

# Streamlit Interface
st.title("🌍 Biodegradable Plastic Waste Forecasting")

st.sidebar.header("📊 Input Parameters")
population = st.sidebar.number_input("Population", min_value=10000, max_value=1000000, value=50000, step=1000)
plastic_consumption_rate = st.sidebar.number_input("Plastic Consumption Rate (kg per person per year)", min_value=1.0, max_value=100.0, value=30.0, step=0.1)
biodegradable_prevalence = st.sidebar.slider("Biodegradable Prevalence (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
biodegradation_rate = st.sidebar.slider("Biodegradation Rate (%)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Predict"):
    prediction = predict_waste(population, plastic_consumption_rate, biodegradable_prevalence, biodegradation_rate)
    st.success(f"Predicted Biodegradable Plastic Waste: {float(prediction):.2f} tons")