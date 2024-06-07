import streamlit as st
import pickle
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Ensure TF-IDF Vectorizer is properly initialized and fitted
# Load your training data
# train_data = pd.read_csv('path_to_training_data.csv')['text_column_name']
# tfidf = TfidfVectorizer()
# tfidf.fit(train_data)

# Or load a pre-fitted TF-IDF Vectorizer if available
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Phishing Detection System")
input_sms = st.text_area("Enter your message here:")

if st.button('View Results'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
