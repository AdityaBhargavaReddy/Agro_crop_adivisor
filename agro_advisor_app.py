import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("crop_dataset.csv")

# Prepare features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌾 Adaptive Crop Forecasting System")

st.write("Enter the following environmental and soil parameters:")

n = st.number_input("Nitrogen (N)", 0, 200, 50)
p = st.number_input("Phosphorous (P)", 0, 200, 50)
k = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

if st.button("Predict Best Crop"):
    input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Recommended Crop: **{prediction.upper()}**")
