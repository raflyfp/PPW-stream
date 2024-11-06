import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Membaca dataset
df = pd.read_csv("preprocessing-kompas.csv")

# Pastikan tidak ada nilai NaN pada kolom 'stopword_removal'
df['stopword_removal'] = df['stopword_removal'].fillna('')

# Mengonversi Data Teks ke TF-IDF
vectorizer = TfidfVectorizer(norm='l2')
tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])

# Menentukan target (y)
y = df['kategori']  # Target adalah kategori berita

# Menambahkan slider untuk memilih jumlah fitur
n_features = st.slider('Pilih jumlah fitur (komponen) yang ingin digunakan:', 
                       min_value=10, max_value=200, step=10, value=100)

# Reduksi Dimensi dengan Truncated SVD berdasarkan jumlah fitur yang dipilih
svd = TruncatedSVD(n_components=n_features, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Pisahkan dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(svd_matrix, y, test_size=0.2, random_state=42)

# Inisialisasi model Regresi Logistik
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Fungsi untuk melakukan prediksi
def predict_category(text_input):
    tfidf_input = vectorizer.transform([text_input])
    svd_input = svd.transform(tfidf_input)
    prediction = model.predict(svd_input)
    return prediction[0]

# Streamlit Layout
st.title("Klasifikasi Kategori Berita")
st.write("Masukkan berita yang ingin dikategorikan untuk mengetahui apakah termasuk kategori Olahraga atau Politik.")

# Input berita dari pengguna
news_text = st.text_area("Masukkan Berita", height=200)

# Tombol untuk submit input
if st.button('Klasifikasikan'):
    if news_text:  # Memastikan ada input
        # Prediksi kategori berita
        predicted_category = predict_category(news_text)
        
        st.subheader("Kategori Berita:")
        st.write(predicted_category)
    else:
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
