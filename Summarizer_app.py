# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine
import numpy as np
import os
import nltk

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  # Downloads the 'punkt' tokenizer data if not already present

# Function to clean text
def cleansing(content):
    content = content.strip()
    content = re.sub(f"[{string.punctuation}]", '', content)
    content = re.sub(r'\d+', '', content)
    content = re.sub(r"\b[a-zA-Z]\b", "", content)
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    content = re.sub(r'\s+', ' ', content)
    return content

# Function to summarize document based on TF-IDF
def summarize_document(melted_tfidf, num_sentences=3):
    tfidf_per_sentence = melted_tfidf.groupby('Sentence')['TF-IDF'].sum()
    top_sentences = tfidf_per_sentence.nlargest(num_sentences)
    summary_sentences = top_sentences.index.tolist()
    return summary_sentences

# Streamlit app interface
st.title("Ringkasan Dokumen")
st.write("Upload a document, and choose the number of sentences for summary.")

# Input text area
document_text = st.text_area("Paste Disini:")

# Slider to choose number of summary sentences
num_sentences = st.slider("Pilih jumlah kalimat:", min_value=1, max_value=10, value=3)

if st.button("Ringkas"):
    if document_text:
        # Step 1: Tokenize sentences
        sentences = sent_tokenize(document_text)
        
        # Step 2: Clean each sentence
        cleaned_sentences = [cleansing(sentence) for sentence in sentences]
        
        # Step 3: Tokenize each sentence into terms
        terms_per_sentence = [word_tokenize(sentence) for sentence in cleaned_sentences]
        
        # Step 4: Combine terms back to string format
        df_terms = pd.DataFrame({'Sentence': sentences, 'Terms_String': [' '.join(terms) for terms in terms_per_sentence]})
        
        # Step 5: Create TF-IDF Vectorizer and transform data
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_terms['Terms_String'])
        
        # Step 6: Create DataFrame for TF-IDF values
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_df['Sentence'] = sentences
        
        # Step 7: Melt the TF-IDF DataFrame
        melted_tfidf = tfidf_df.melt(id_vars=['Sentence'], var_name='Term', value_name='TF-IDF')
        melted_tfidf = melted_tfidf[melted_tfidf['TF-IDF'] != 0]
        
        # Step 8: Summarize the document
        summary = summarize_document(melted_tfidf, num_sentences=num_sentences)
        
        # Display the summary
        st.write("## Document Summary")
        for idx, sentence in enumerate(summary, 1):
            st.write(f"{idx}. {sentence}")
    else:
        st.warning("Please paste a document text to summarize.")
