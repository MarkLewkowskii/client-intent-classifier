import streamlit as st
import joblib

clf = joblib.load("model/model.pkl")
vec = joblib.load("model/vectorizer.pkl")

st.title("Klasyfikator zgłoszeń klienta")

text = st.text_area("Wpisz treść zgłoszenia:")
if st.button("Klasyfikuj"):
    X = vec.transform([text])
    pred = clf.predict(X)[0]
    st.success(f"Kategoria: {pred}")
