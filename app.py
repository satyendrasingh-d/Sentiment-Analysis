import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text) # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"\d+", "", text)     # Remove numbers
    text = text.translate(str.maketrans("","", string.punctuation)) # Remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Cache the models so they only load once when the app starts
@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Load our saved models
model, vectorizer = load_models()

# --- STREAMLIT UI DESIGN ---

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠")

st.title("🧠 Tweet Sentiment Analyzer")
st.write("Enter text below to determine if the sentiment is Positive or Negative.")

# Text input area
user_input = st.text_area("What's on your mind?", placeholder="Type your sentence here...")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            # 1. Clean the input
            cleaned_text = clean_text(user_input)
            
            # 2. Vectorize
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # 3. Predict
            prediction = model.predict(text_vectorized)[0]
            
            # Note: The Sentiment140 dataset usually maps 0 to Negative and 4 to Positive. 
            # Adjust these numbers if your specific subset uses 0 and 1.
            if prediction == 4 or prediction == 1:
                st.success("### Prediction: Positive 😊")
            else:
                st.error("### Prediction: Negative 😔")
                
# Optional: Display the cleaned text to show how the model sees it
with st.expander("Show me how the model sees my text (Under the hood)"):
    if user_input:
        st.code(clean_text(user_input))
