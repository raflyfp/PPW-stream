import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Muat TfidfVectorizer dan model Regresi Logistik yang telah disimpan
@st.cache(allow_output_mutation=True)
def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('logistic_regression_model.joblib')
    return vectorizer, model

vectorizer, model = load_model()

# Layout Aplikasi Streamlit
st.title("Klasifikasi Kategori Berita")
st.write("Masukkan konten berita di bawah ini, dan model akan memprediksi kategorinya.")

# Input Teks
user_input = st.text_area("Konten Berita", "Ketik konten berita Anda di sini...", height=200)

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks untuk diklasifikasikan.")
    else:
        # Transformasi input menggunakan vectorizer yang dimuat
        input_tfidf = vectorizer.transform([user_input])

        # Melakukan prediksi
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)

        # Menampilkan hasil
        st.success(f"**Kategori yang Diprediksi:** {prediction[0]}")
        
        # Opsional: Menampilkan probabilitas prediksi
        # st.write("**Probabilitas Prediksi:**")
        # proba_df = pd.DataFrame({
        #     'Kategori': model.classes_,
        #     'Probabilitas': prediction_proba[0]
        # }).sort_values(by='Probabilitas', ascending=False)
        # st.write(proba_df)
