import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import time
import numpy as np

# Load dataset
df = pd.read_csv("crop_dataset.csv")

# Prepare features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Calculate model accuracy over multiple splits to find min, max, avg
accuracies = []
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    temp_model = RandomForestClassifier()
    temp_model.fit(X_train, y_train)
    y_pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

min_accuracy = np.min(accuracies)
max_accuracy = np.max(accuracies)
avg_accuracy = np.mean(accuracies)

# Train final model on full dataset
model = RandomForestClassifier()
model.fit(X, y)

# Crop diseases dictionary
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

# Translations dictionary
translations = {
    "title": {
        "en": "🌾 Adaptive Crop Forecasting System",
        "hi": "🌾 अनुकूली फसल पूर्वानुमान प्रणाली",
        "te": "🌾 అనుకూల పంట అంచనా వ్యవస్థ"
    },
    "input_label": {
        "en": "Enter the following environmental and soil parameters:",
        "hi": "निम्नलिखित पर्यावरण और मृदा पैरामीटर दर्ज करें:",
        "te": "క్రింది పర్యావరణ మరియు మట్టి పరామితులను నమోదు చేయండి:"
    },
    "predict_btn": {
        "en": "Predict Best Crop",
        "hi": "सर्वश्रेष्ठ फसल का पूर्वानुमान लगाएं",
        "te": "ఉత్తమ పంటను అంచనా వేయండి"
    },
    "top_recommendations": {
        "en": "🌟 Top 5 Recommended Crops",
        "hi": "🌟 शीर्ष 5 अनुशंसित फसलें",
        "te": "🌟 టాప్ 5 సూచించిన పంటలు"
    },
    "chance": {
        "en": "Chance",
        "hi": "संभावना",
        "te": "సాధ్యత"
    },
    "possible_diseases": {
        "en": "Possible Future Diseases",
        "hi": "संभावित भविष्य की बीमारियाँ",
        "te": "సంభావ్య భవిష్యత్ వ్యాధులు"
    },
    "accuracy": {
        "en": "Model Accuracy Overview",
        "hi": "मॉडल सटीकता अवलोकन",
        "te": "మోడల్ ఖచ్చితత్వం అవలోకనం"
    },
    "loading_text": {
        "en": "Analyzing your input and calculating best crop recommendations...",
        "hi": "आपके इनपुट का विश्लेषण और सर्वोत्तम फसल की सिफारिशें गणना कर रहा है...",
        "te": "మీ ఇన్‌పుట్‌ను విశ్లేషిస్తూ ఉత్తమ పంట సిఫారసులను లెక్కిస్తున్నాం..."
    },
    "interpretation": {
        "en": """**How to interpret:**
- The pie chart shows likelihood of each crop.
- The bar chart compares probabilities.
- Each crop card lists possible diseases.
- Accuracy based on historical data.""",
        "hi": """**कैसे समझें:**
- पाई चार्ट में प्रत्येक फसल की संभावना दिखती है।
- बार चार्ट संभावनाओं की तुलना करता है।
- प्रत्येक फसल कार्ड में संभावित बीमारियाँ हैं।
- सटीकता ऐतिहासिक डेटा पर आधारित है।""",
        "te": """**వివరణ:**
- పై చార్ట్ ప్రతి పంట అవకాశాన్ని చూపిస్తుంది.
- బార్ చార్ట్ అవకాశాలను పోల్చుతుంది.
- ప్రతి పంట కార్డ్‌లో వ్యాధుల జాబితా ఉంటుంది.
- ఖచ్చితత్వం చారిత్రక డేటాపై ఆధారపడి ఉంటుంది."""
    },
    "success_msg": {
        "en": "Use these insights to make informed decisions and monitor diseases to maximize yield! 🌱",
        "hi": "इन जानकारियों का उपयोग करके सूचित निर्णय लें और फसलों की देखभाल करें! 🌱",
        "te": "ఈ జ్ఞానాన్ని ఉపయోగించి బుద్ధిమంతమైన నిర్ణయాలు తీసుకోండి మరియు పంటల పర్యవేక్షణ చేయండి! 🌱"
    }
}

# Language selector
lang = st.selectbox("Choose Language / भाषा चुनें / భాషను ఎంచుకోండి", ["English", "Hindi", "Telugu"])
lang_key = {"English": "en", "Hindi": "hi", "Telugu": "te"}[lang]
T = lambda key: translations[key][lang_key]

# UI
st.title(T("title"))
st.write(T("input_label"))

# Inputs
n = st.number_input("Nitrogen (N)", 0, 200, 50)
p = st.number_input("Phosphorous (P)", 0, 200, 50)
k = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

if st.button(T("predict_btn")):
    progress_text = T("loading_text")
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

    st.markdown(f"<h2 style='color:#4CAF50;'>{T('top_recommendations')}</h2>", unsafe_allow_html=True)

    fig = px.pie(
        names=crops, values=probs,
        title="Top 5 Crop Recommendation Probabilities",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0.05, 0, 0, 0])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h4 style='color:#2196F3;'>Probability Bar Chart</h4>", unsafe_allow_html=True)
    st.bar_chart({"Crop": crops, "Probability": probs})

    for i, (crop, prob) in enumerate(zip(crops, probs)):
        with st.container():
            st.markdown(f"""
                <div style='background: linear-gradient(90deg, #e3ffe8 0%, #f9f9f9 100%);
                            border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 8px #b2f7ef;'>
                <h3 style='color:#388e3c;'>{i+1}. {crop.title()}</h3>
                <b>{T('chance')}:</b> <span style='color:#d84315;font-size:18px'>{prob*100:.2f}%</span><br>
                <b>{T('possible_diseases')}:</b>
            """, unsafe_allow_html=True)
            diseases = crop_diseases.get(crop, ["No data available"])
            for disease in diseases:
                st.markdown(f"- {disease}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<h4 style='color:#6a1b9a;'>{T('accuracy')}:</h4>", unsafe_allow_html=True)
    st.write(f"🔽 Min Accuracy: {min_accuracy*100:.2f}%")
    st.write(f"🔼 Max Accuracy: {max_accuracy*100:.2f}%")
    st.write(f"📉 Avg Accuracy: {avg_accuracy*100:.2f}%")

    st.info(T("interpretation"))
    st.success(T("success_msg"))
