import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import plotly.express as px
import plotly.graph_objects as go
import time

# Load dataset
df = pd.read_csv("crop_dataset.csv")

# 📊 Accuracy Calculation
accuracies = []
for i in range(10):
    df_shuffled = shuffle(df, random_state=i)
    X = df_shuffled[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df_shuffled['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    temp_model = RandomForestClassifier()
    temp_model.fit(X_train, y_train)
    y_pred = temp_model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

model = temp_model
accuracy = round(sum(accuracies) / len(accuracies), 4)
min_accuracy = round(min(accuracies), 4)
max_accuracy = round(max(accuracies), 4)

# 🌐 Language selector
language = st.sidebar.selectbox("🌍 Select Language / भाषा चुनें / భాషను ఎంచుకోండి",
                                 ["English", "Hindi (हिंदी)", "Telugu (తెలుగు)"])

texts = {
    "English": {
        "title": "🌾 Adaptive Crop Forecasting System",
        "input_section": "Enter the following environmental and soil parameters:",
        "predict_button": "Predict Best Crop",
        "loading": "Analyzing your input...",
        "recommendations": "🌿 Recommendations",
        "top_crops": "🌟 Top 5 Recommended Crops",
        "chart_title": "Top Crop Probabilities",
        "bar_chart": "📊 Probability Comparison",
        "chance": "Chance",
        "future_diseases": "🦠 Future Diseases & Treatments",
        "model_accuracy": "✅ Model Accuracy",
        "footer": "📈 Make informed farming decisions based on these insights!",
        "soil_fertility": "🌱 Soil Fertility Score",
        "fertility_description": "An overall indicator based on N, P, K, and pH"
    }
}

txt = texts[language]

# 🧪 Disease to Pesticide Mapping
disease_pesticides = {
    "Bacterial Leaf Blight": "Copper-based fungicides or Streptomycin",
    "Blast": "Tricyclazole or Carbendazim",
    "Brown Spot": "Mancozeb or Chlorothalonil",
    "Downy Mildew": "Metalaxyl or Mancozeb",
    "Leaf Blight": "Zineb or Mancozeb",
    "Rust": "Propiconazole or Mancozeb",
    "Ascochyta Blight": "Chlorothalonil or Captan",
    "Fusarium Wilt": "Carbendazim or Trichoderma spp.",
    "Anthracnose": "Copper fungicides or Carbendazim",
    "Sterility Mosaic Disease": "No cure, manage with vector control (mites)",
    "Phytophthora Blight": "Metalaxyl or Fosetyl-Al",
    "Yellow Mosaic Virus": "No cure, control with insecticides (Imidacloprid)",
    "Powdery Mildew": "Sulfur or Myclobutanil",
    "Cercospora Leaf Spot": "Chlorothalonil or Thiophanate-methyl",
    "Apple Scab": "Captan or Mancozeb",
    "Citrus Canker": "Copper oxychloride spray",
    "Greening Disease": "No cure, control psyllid vector (Imidacloprid)",
    "Papaya Ringspot Virus": "No treatment; use virus-free planting material",
    "Bud Rot": "Copper oxychloride or Metalaxyl",
    "Stem Bleeding": "Hexaconazole + neem cake",
    "Bacterial Blight": "Copper hydroxide or Streptomycin",
    "Wilt": "Carbendazim or Trichoderma viride",
    "Coffee Leaf Rust": "Copper-based fungicides",
    "Coffee Berry Disease": "Chlorothalonil or Copper oxychloride"
}

# 🦠 Crop disease info
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

# 🧮 Inputs
st.title(txt["title"])
st.write(txt["input_section"])

n = st.number_input("Nitrogen (N)", 0, 200, 50)
p = st.number_input("Phosphorous (P)", 0, 200, 50)
k = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

# 🌱 Soil Fertility Calculator
def calculate_soil_fertility(n, p, k, ph):
    n_score = min(n / 120, 1.0)
    p_score = min(p / 100, 1.0)
    k_score = min(k / 120, 1.0)
    ph_score = 1 - abs(ph - 6.5) / 3.5
    fertility_score = (0.3 * n_score + 0.3 * p_score + 0.3 * k_score + 0.1 * ph_score)
    return round(fertility_score * 100, 2)

# 🎯 Predict and Display
if st.button(txt["predict_button"]):
    progress_bar = st.progress(0, text=txt["loading"])
    for i in range(0, 101, 10):
        time.sleep(0.03)
        progress_bar.progress(i, text=f"{txt['loading']} ({i}%)")
    progress_bar.empty()

    input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
    proba = model.predict_proba(input_data)[0]
    crop_indices = proba.argsort()[::-1][:5]
    crops = model.classes_[crop_indices]
    probs = proba[crop_indices]

    with st.expander(txt["recommendations"], expanded=True):
        st.markdown(f"<h2 style='color:#388e3c;'>{txt['top_crops']}</h2>", unsafe_allow_html=True)

        fig = px.pie(names=crops, values=probs, title=txt["chart_title"], color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0.05, 0, 0, 0])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"<h4 style='color:#1976d2;'>{txt['bar_chart']}</h4>", unsafe_allow_html=True)
        st.bar_chart({"Crop": crops, "Probability": probs})

        for i, (crop, prob) in enumerate(zip(crops, probs)):
            diseases = crop_diseases.get(crop, ["No data available"])
            st.markdown(f"""
                <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; margin-bottom:12px;'>
                    <h3 style='color:#1b5e20;'>{i+1}. {crop.title()}</h3>
                    <b style='color:#0d47a1;'>{txt['chance']}:</b> <span style='color:#d84315;font-size:18px'>{prob*100:.2f}%</span>
                    <h4 style='color:#b71c1c; margin-top:10px;'>{txt['future_diseases']}</h4>
                    <ul style='color:#37474f; font-size:15px;'>
                        {''.join(f"<li><b>{disease}</b>: {disease_pesticides.get(disease, 'No treatment info')}</li>" for disease in diseases)}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # 🌱 Soil Fertility Score
        fertility_score = calculate_soil_fertility(n, p, k, ph)
        st.markdown(f"<h3 style='color:#4caf50;'>{txt['soil_fertility']}</h3>", unsafe_allow_html=True)
        st.caption(txt["fertility_description"])
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fertility_score,
            title={'text': "Fertility Score", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#43a047"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcdd2"},
                    {'range': [40, 70], 'color': "#fff59d"},
                    {'range': [70, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': fertility_score}
            }
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # ✅ Accuracy
        st.markdown(f"""
        <h4 style='color:#6a1b9a;'>{txt['model_accuracy']}:</h4>
        <ul style='color:#388e3c; font-size:16px;'>
          <li><b>Average Accuracy:</b> {accuracy*100:.2f}%</li>
          <li><b>Max Accuracy:</b> {max_accuracy*100:.2f}%</li>
          <li><b>Min Accuracy:</b> {min_accuracy*100:.2f}%</li>
        </ul>
        """, unsafe_allow_html=True)

        st.success(txt["footer"])
