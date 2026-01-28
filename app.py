import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import re

# ==========================================================
# 1. KONFIGURASI HALAMAN (UI Dashboard)
# ==========================================================
st.set_page_config(page_title="Bilingual Sentiment & Integrity Dashboard", layout="wide")

# ==========================================================
# 2. LOAD MODEL (Hugging Face Transformers)
# ==========================================================
@st.cache_resource
def load_models():
    # Model Sentiment: Menyokong pelbagai bahasa (Malay & English)
    sentiment_model = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    
    # Model Emotion: Untuk mengesan perasaan tersembunyi (Joy, Anger, Fear, etc.)
    emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
    
    return sentiment_model, emotion_model

sentiment_pipe, emotion_pipe = load_models()

# ==========================================================
# 3. ANTARAMUKA PENGGUNA (Header & Input)
# ==========================================================
st.title("📊 NLP Sentiment & Review Integrity Dashboard")
st.markdown("""
Sistem ini menganalisis sentimen dan emosi dalam **Bahasa Melayu & Inggeris**. 
Ia juga menyemak sama ada *Rating* yang diberikan konsisten dengan teks ulasan.
""")

with st.container():
    col_input1, col_input2 = st.columns([1, 3])
    with col_input1:
        # Input Rating untuk semak konsistensi 
        user_rating = st.slider("Select User Rating (1-5):", 1, 5, 3)
    
    with col_input2:
        # Input Teks/Ayat
        user_text = st.text_area("Enter your sentences:", placeholder="Contoh: Makanan sedap tapi harga agak mahal...")

# ==========================================================
# 4. LOGIK PEMPROSESAN (Apabila Butang Ditekan)
# ==========================================================
if st.button("Start Deep Analysis"):
    if user_text:
        # A. Pecahkan ayat (Sentence Tokenization) untuk ketepatan "Mixed Feelings"
        sentences = re.split(r'(?<=[.!?])\s+', user_text.strip())
        
        sentence_data = []
        for s in sentences:
            if len(s.strip()) > 2:
                res = sentiment_pipe(s)[0]
                sentence_data.append({"Sentence": s, "Label": res['label'], "Score": res['score']})
        
        df_sent = pd.DataFrame(sentence_data)

        # B. Analisis Emosi Keseluruhan (Hidden Feelings)
        emotions_raw = emotion_pipe(user_text)[0]
        df_emot = pd.DataFrame(emotions_raw)

        # ==========================================================
        # 5. PAPARAN KEPUTUSAN (DASHBOARD)
        # ==========================================================
        st.divider()
        
        # Bahagian 1: Integrity Check (Rating vs Sentiment)
        # Ambil sentimen dominan dari input
        final_sentiment = df_sent['Label'].mode()[0] 
        
        st.subheader("🛡️ Review Integrity Status")
        is_consistent = False
        if final_sentiment == "positive" and user_rating >= 4: is_consistent = True
        elif final_sentiment == "negative" and user_rating <= 2: is_consistent = True
        elif final_sentiment == "neutral" and user_rating == 3: is_consistent = True
        
        if is_consistent:
            st.success(f"CONSISTENT: The {user_rating}-star rating matches the {final_sentiment} sentiment.")
        else:
            st.error(f"MISMATCHED: The user gave {user_rating} stars, but the system detected {final_sentiment} sentiment.")

        # Bahagian 2: Sentence-Level Analysis & Emotion Chart
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.subheader("📝 Breakdown per Sentence")
            for _, row in df_sent.iterrows():
                color = "green" if row['Label'] == 'positive' else "red" if row['Label'] == 'negative' else "gray"
                st.write(f"**Text:** {row['Sentence']}")
                st.markdown(f"**Sentiment:** :{color}[{row['Label'].upper()}] (Confidence: {row['Score']:.2%})")
                st.write("---")

        with col_res2:
            st.subheader("🎭 Emotional Intensity")
            # Visualisasi bar chart untuk emosi
            fig = px.bar(df_emot, x='score', y='label', orientation='h', color='label',
                         labels={'label': 'Emotion', 'score': 'Intensity'})
            st.plotly_chart(fig, use_container_width=True)

        # Bahagian 3: Kesimpulan Akhir
        st.subheader("📌 Summary Explanation")
        if "positive" in df_sent['Label'].values and "negative" in df_sent['Label'].values:
            st.warning("Detection: This input contains **Mixed Feelings**. The user expresses both satisfaction and dissatisfaction.")
        elif final_sentiment == "positive":
            st.success("Detection: The overall feedback is positive and constructive.")
        else:
            st.error("Detection: The overall feedback is negative and requires attention.")
            
    else:
        st.warning("Please enter some text to analyze.")
