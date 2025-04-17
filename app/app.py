import streamlit as st
import joblib
import numpy as np

# Завантаження моделі та векторизатора
clf = joblib.load("model/model.pkl")
vec = joblib.load("model/vectorizer.pkl")

# --------------- ІНТЕРФЕЙС ---------------
st.set_page_config(page_title="Klasyfikator zgłoszeń", page_icon="💬", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FAFAFA;'>💬 Klasyfikator zgłoszeń klienta</h1>", unsafe_allow_html=True)
st.write("Wpisz treść zgłoszenia klienta, a model przypisze mu odpowiednią kategorię.")

# --------------- ІСТОРІЯ ---------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------- ВВІД ---------------
user_input = st.text_area("✉️ Wiadomość", placeholder="Np. Chciałbym otrzymać fakturę za zamówienie...")

if st.button("🔍 Klasyfikuj", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Wpisz wiadomość do klasyfikacji.")
    else:
        X = vec.transform([user_input])
        pred = clf.predict(X)[0]
        probas = clf.predict_proba(X)[0]
        classes = clf.classes_

        # Додаємо до історії
        st.session_state.history.append((user_input, pred))

        st.success(f"📌 **Kategoria:** `{pred}`")

        st.subheader("📊 Prawdopodobieństwa:")
        for label, p in zip(classes, probas):
            st.write(f"- `{label}`: **{p:.2%}**")

# --------------- ІСТОРІЯ ВИВОДУ ---------------
if st.session_state.history:
    st.divider()
    st.subheader("🕓 Historia tej sesji:")
    for i, (text, label) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}.** _{text}_ → `🧠 {label}`")
