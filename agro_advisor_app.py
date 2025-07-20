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

# Calculate model accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_acc = RandomForestClassifier()
model_acc.fit(X_train, y_train)
y_pred = model_acc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

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

# Static mapping of crops to possible future diseases (example, can be expanded)
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
    # Add more crops and diseases as needed
}

if st.button("Predict Best Crop"):
    # Show a loading bar with percentage
    progress_text = "Analyzing your input and calculating best crop recommendations..."
    progress_bar = st.progress(0, text=progress_text)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(percent_complete, text=f"{progress_text} ({percent_complete}%)")
    progress_bar.empty()

    input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
    proba = model.predict_proba(input_data)[0]
    crop_indices = proba.argsort()[::-1][:5]
    crops = model.classes_[crop_indices]
    probs = proba[crop_indices]

    # Try to use st.modal for a pop-up effect (Streamlit >=1.32), fallback to expander if not available
    try:
        modal = st.modal("🎉 See Your Crop Recommendations & Insights! (Click to Close)", key="modal1")
    except AttributeError:
        modal = st.expander("🎉 See Your Crop Recommendations & Insights! (Click to Expand)", expanded=True)

    with modal:
        st.markdown("<h2 style='color:#4CAF50;'>🌟 Top 5 Recommended Crops</h2>", unsafe_allow_html=True)
        st.write("Based on your input, here are the best crop options and their likelihood:")

        # Pie chart with Plotly for super visual effect
        fig = px.pie(names=crops, values=probs, title="Top 5 Crop Recommendation Probabilities", color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0.05, 0, 0, 0])
        st.plotly_chart(fig, use_container_width=True)

        # Stunning bar chart
        st.markdown("<h4 style='color:#2196F3;'>Probability Bar Chart</h4>", unsafe_allow_html=True)
        st.bar_chart({"Crop": crops, "Probability": probs})

        # Beautiful cards for each crop using st.container for new render component
        for i, (crop, prob) in enumerate(zip(crops, probs)):
            with st.container():
                st.markdown(f"<div style='background: linear-gradient(90deg, #e3ffe8 0%, #f9f9f9 100%); border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 8px #b2f7ef;'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:#388e3c;'>{i+1}. {crop.title()}</h3>", unsafe_allow_html=True)
                st.markdown(f"<b>Chance:</b> <span style='color:#d84315;font-size:18px'>{prob*100:.2f}%</span>", unsafe_allow_html=True)
                diseases = crop_diseases.get(crop, ["No data available, but common fungal, bacterial, and viral diseases may occur. Monitor regularly for leaf spots, wilts, and blights."])
                st.markdown(f"<b>Possible Future Diseases:</b>", unsafe_allow_html=True)
                for disease in diseases:
                    st.markdown(f"- {disease}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<h4 style='color:#6a1b9a;'>Model Accuracy: <span style='color:#43a047'>{accuracy*100:.2f}%</span></h4>", unsafe_allow_html=True)
        st.info("""
        **How to interpret:**
        - The pie chart shows the likelihood of each crop being the best fit for your conditions.
        - The bar chart gives a quick comparison of probabilities.
        - Each crop card lists possible future diseases to watch for.
        - Model accuracy is based on historical data and may vary with real-world conditions.
        """)
        st.success("Use these insights to make informed decisions and monitor for the listed diseases to maximize your yield! 🌱")
