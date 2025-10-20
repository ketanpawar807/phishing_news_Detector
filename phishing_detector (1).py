import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("🛡️ Phishing Email Detector")
st.write("Enter an email text below to check if it might be a phishing attempt!")

@st.cache_data
def load_data():
    df = pd.read_csv("emails.csv")
    return df

data = load_data()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{acc*100:.2f}%**")

user_input = st.text_area("✉️ Paste email text here:")

if st.button("Analyze"):
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vec)[0]
    if prediction == "phishing":
        st.error("🚨 This email looks **phishing or suspicious!**")
    else:
        st.success("✅ This email seems **legitimate.**")

st.info("This tool is for educational awareness about email phishing.")
