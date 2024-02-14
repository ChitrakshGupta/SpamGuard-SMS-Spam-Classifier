import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.data.path.append("nltk_data")





tfidf=pickle.load(open("vectorizer.pkl",'rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/Sms Spam Classifier")

input_sms=st.text_area("Enter the message")


def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenization :- convert sentence in words list
    # removing special char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    # removing stop words in puncutaion
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))  # remove s , ing
    text = y[:]
    y.clear()
    return " ".join(text)

if st.button("Predict"):
    #1. preprocess
    transformed_sms=transform_text(input_sms)
    #2. vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3.predict
    result=model.predict(vector_input)[0]
    #4.display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
footer = """
---
Developed by [Chitraksh](https://www.linkedin.com/in/chitraksh7/)
"""

# Add footer to the app
st.markdown(footer, unsafe_allow_html=True)
