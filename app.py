import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
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

# Load the training data
# Uncomment and modify the next line if you need to fit the vectorizer
# train_data = pd.read_csv('path_to_training_data.csv')

# Initialize and fit the TF-IDF vectorizer if it's not already fitted
# Uncomment the next lines if you are fitting the vectorizer here
# tfidf = TfidfVectorizer()
# tfidf.fit(train_data['text_column_name'])  # Make sure to replace 'text_column_name' with your actual text column

# Save the fitted vectorizer (optional, only if you're fitting it here)
# Uncomment the next line if you need to save the fitted vectorizer
# pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# Load the fitted vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Load the classification model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Phishing Detection System")
input_sms = st.text_area("Enter your message here:")

if st.button('View Results'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed input
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the loaded model
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
