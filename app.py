import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # [:] THIS IS CLONING AND USE FOR COPY FROM [start : end].
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # stemming  dancing-> danc, loving-> love
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
background_image = 'picture.jpeg'
st.image(background_image, width=15000)

# input_sms = st.text_input("Enter your message")
input_sms = st.text_area("Enter your email message")

st.write("Your message:")
st.write(input_sms)

if st.button('Predict'):
# 1. preprocess
    transformed_sms = transform_text(input_sms)
# 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
# 3. prdict
    result = model.predict(vector_input)[0]
# 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam ")


# Add a bold line above the footer
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# Footer content
st.write("Copy© 2026 M.Athar | Made With Muhammad Athar Ur Rahman")