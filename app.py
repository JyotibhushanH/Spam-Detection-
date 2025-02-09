import streamlit as st
import numpy as np
import joblib  # For loading ML models
import re
import string

# ✅ FIX: Page Config MUST be the first Streamlit command
st.set_page_config(page_title="📩 SMS Spam Detector", layout="wide")

# ✅ Load Model and Vectorizer
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("model.pkl")  # Load trained spam classifier
        vectorizer = joblib.load("vectorizer.pkl")  # Load TF-IDF vectorizer
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_resources()

# ✅ Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# ✅ Streamlit UI
st.title("📩 SMS Spam Detection App")
st.write("Enter an SMS message below to check if it's spam or not.")

# ✅ User Input
user_input = st.text_area("Enter SMS:", "")

# ✅ Prediction Button
if st.button("Check Spam"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        
        if model and vectorizer:
            # ✅ FIX: Convert input to TF-IDF vector before prediction
            input_tfidf = vectorizer.transform([processed_text])  # Now a 2D array
            prediction = model.predict(input_tfidf)
            result = "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"
            st.success(f"Prediction: **{result}**")
        else:
            st.error("Model or vectorizer not found! Ensure 'model.pkl' and 'vectorizer.pkl' are present.")
    else:
        st.warning("Please enter a message.")

# ✅ Debugging Output
st.write("🔹 Model Loaded Successfully.")
