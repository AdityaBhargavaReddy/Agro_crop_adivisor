import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import time

# Load dataset
df = pd.read_csv("crop_dataset.csv")

# Prepare features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Accuracy calculation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_acc = RandomForestClassifier()
model_acc.fit(X_train, y_train)
y_pred = model_acc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# UI
st.title("🌾 Adaptive Crop Forecasting System")
st.write("Enter the following environmental and soil parameters:")

n = st.number_input("Nitrogen (N)", 0, 200, 50)
p = st.number_input("Phosphorous (P)", 0, 200, 50)
k = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

# Future diseases mapping
crop_diseases = {
    "rice": ["Bacterial Leaf Blight", "Blast", "Brown Spot"],
    "maize": ["Downy Mildew", "Leaf Blight", "Rust"],
    "chickpea": ["Ascochyta Blight", "Fusarium Wilt"],
    "kidneybeans": ["Anthracnose", "Rust"],
    "pigeonpeas": ["Sterility Mosaic Disease", "Phytophthora Blight"],
    "mothbeans": ["Yellow Mosaic Virus", "Powdery Mildew"],
    "mungbean": ["Cercospora Leaf Spot", "Yellow Mosaic Virus"],
    "blackgram": ["Cercospora Leaf Spot", "Powdery Mildew"],
    "lentil": ["Rust", "Ascochyta Blight"],
    "banana": ["Panama Disease", "Sigatoka Leaf Spot"],
    "mango": ["Powdery Mildew", "Anthracnose"],
    "grapes": ["Downy Mildew", "Powdery Mildew"],
    "watermelon": ["Anthracnose", "Fusarium Wilt"],
    "muskmelon": ["Powdery Mildew", "Downy Mildew"],
    "apple": ["Apple Scab", "Powdery Mildew"],
    "orange": ["Citrus Canker", "Greening Disease"],
    "papaya": ["Papaya Ringspot Virus", "Powdery Mildew"],
    "coconut": ["Bud Rot", "Stem Bleeding"],
    "cotton": ["Bacterial Blight", "Wilt"],
    "jute": ["Stem Rot", "Anthracnose"],
    "coffee": ["Coffee Leaf Rust", "Coffee Berry Disease"],
}

if st.button("Predict Best Crop"):
    progress_text = "🔍 Analyzing your input..."
    progress_bar = st.progress(0, text=progress_text)
    for i in range(0, 101, 10):
        time.sleep(0.03)
        progress_bar.progress(i, text=f"{progress_text} ({i}%)")
    progress_bar.empty()

    input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
    proba = model.predict_proba(input_data)[0]
    crop_indices = proba.argsort()[::-1][:5]
    crops = model.classes_[crop_indices]
    probs = proba[crop_indices]

    try:
        modal = st.modal("🌿 Recommendations (Click to Close)", key="modal1")
    except AttributeError:
        modal = st.expander("🌿 Recommendations (Click to Expand)", expanded=True)

    with modal:
        st.markdown("<h2 style='color:#2e7d32;'>🌟 Top 5 Recommended Crops</h2>", unsafe_allow_html=True)
        st.write("Based on your input, here are the best crop options:")

        # Pie chart
        fig = px.pie(names=crops, values=probs, title="Top Crop Probabilities", color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0.05, 0, 0, 0])
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        st.markdown("<h4 style='color:#1565c0;'>📊 Probability Comparison</h4>", unsafe_allow_html=True)
        st.bar_chart({"Crop": crops, "Probability": probs})

        for i, (crop, prob) in enumerate(zip(crops, probs)):
            with st.container():
                st.markdown(f"<div style='background: linear-gradient(90deg, #f1f8e9, #e8f5e9); border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 8px #c8e6c9;'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:#388e3c;'>{i+1}. {crop.title()}</h3>", unsafe_allow_html=True)
                st.markdown(f"<b>Chance:</b> <span style='color:#d84315;font-size:18px'>{prob*100:.2f}%</span>", unsafe_allow_html=True)

                # 🦠 Future Diseases Heading
                st.markdown("<h4 style='color:#e65100;'>🦠 Future Diseases</h4>", unsafe_allow_html=True)
                diseases = crop_diseases.get(crop, ["No data available. Monitor for common blights, wilts, or fungal infections."])
                for disease in diseases:
                    st.markdown(f"- {disease}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<h4 style='color:#6a1b9a;'>✅ Model Accuracy: <span style='color:#43a047'>{accuracy*100:.2f}%</span></h4>", unsafe_allow_html=True)
        st.info("""
        - The pie and bar charts show how confident the model is about the crop predictions.
        - Use disease insights to take preventive measures.
        - Results are based on your input and historical crop data.
        """)
        st.success("📈 Make informed farming decisions based on these insights! 🌱")
