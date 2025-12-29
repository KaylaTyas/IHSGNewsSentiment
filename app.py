import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine, text
from datetime import date

# Config
st.set_page_config(
    page_title="IHSG Prediction Dashboard",
    page_icon="🔮",
    layout="wide"
)

engine = create_engine(os.getenv("DB_URL"))
TODAY = pd.to_datetime(date.today())

# Custom CSS
st.markdown("""
<style>
    .big1-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #0072B5 0%, #00D4FF 100%);
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        margin: 1rem 0;
    }
            
    .big2-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #A605FC 0%, #FF0088 100%);
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 0, 136, 0.3);
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-date {
        font-size: 0.95rem;
        font-weight: 600;
        color: #888;
        text-align: center;
    }
    .news-card {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        border: 2px solid #FF0088 !important;
        background: linear-gradient(135deg, #FF0088 0%, #A605FC 100%)!important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 15px rgba(255, 0, 136, 0.15) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3399 0%, #b833ff 100%) !important;
        border-color: #FF0088 !important;
        box-shadow: 0 8px 25px rgba(255, 0, 136, 0.5) !important;
        transform: translateY(-2px);
        color: white !important;
    }
</style>""", unsafe_allow_html=True)

# Load data function
@st.cache_data(ttl=600)
def load_today_news():
    """Get today's news, fallback to yesterday if empty"""
    query = """
    SELECT
        r.tanggal,
        r.judul,
        r.url,
        p.sentiment_label,
        p.sentiment_score
    FROM raw_news r
    JOIN processed_news p ON p.raw_id = r.id
    WHERE r.tanggal = CURRENT_DATE
    ORDER BY p.sentiment_score DESC"""
    df = pd.read_sql(query, engine)
    
    if df.empty:
        query_yesterday = """
        SELECT
            r.tanggal,
            r.judul,
            r.url,
            p.sentiment_label,
            p.sentiment_score
        FROM raw_news r
        JOIN processed_news p ON p.raw_id = r.id
        WHERE r.tanggal = CURRENT_DATE - INTERVAL '1 day'
        ORDER BY p.sentiment_score DESC"""
        df = pd.read_sql(query_yesterday, engine)
    
    return df

@st.cache_data(ttl=600)
def load_prediction():
    query = """
    SELECT date, predicted_close, predicted_pct
    FROM predictions
    ORDER BY date DESC
    LIMIT 1"""
    return pd.read_sql(query, engine)

@st.cache_data(ttl=600)
def load_weekly_data():
    """Get last 6 days from prices + tomorrow's prediction (SINGLE LINE)"""
    query_prices = """
    SELECT 
        "Tanggal" as date_str,
        "Terakhir" as close
    FROM prices"""
    df_prices = pd.read_sql(query_prices, engine)
    
    df_prices['date'] = pd.to_datetime(df_prices['date_str'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    df_prices = df_prices.dropna(subset=['date'])
    
    df_prices = df_prices.sort_values('date', ascending=False).head(6)
    df_prices = df_prices.sort_values('date')
    
    df_prices['close'] = df_prices['close'].astype(str).str.replace(',', '').str.replace('.', '')
    df_prices['close'] = pd.to_numeric(df_prices['close'], errors='coerce') / 100
    
    df_prices['type'] = 'actual'
    
    query_pred = """
    SELECT date, predicted_close as close
    FROM predictions
    ORDER BY date DESC
    LIMIT 1"""
    df_pred = pd.read_sql(query_pred, engine)
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_pred['type'] = 'prediction'
    
    df_combined = pd.concat([df_prices[['date', 'close', 'type']], df_pred], ignore_index=True)
    df_combined = df_combined.sort_values('date')
    return df_combined

@st.cache_data(ttl=600)
def load_today_ihsg():
    query = """
    SELECT 
        "Tanggal" as date_str,
        "Terakhir" as close
    FROM prices"""
    df = pd.read_sql(query, engine)
    
    df['date'] = pd.to_datetime(df['date_str'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    
    df = df.sort_values('date', ascending=False).head(1)
    
    df['close'] = df['close'].astype(str).str.replace('.', '').str.replace(',', '.')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df

st.markdown("""
    <style>
    .target {
        display: block;
        position: relative;
        top: -100px;
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<a class="target" id="top_tabs"></a>', unsafe_allow_html=True)

# TABS
tab1, tab2 = st.tabs(["🏠 Home", "📊 Dashboard"])

# TAB 1: HOME
with tab1:
    st.title("📰 IHSG Prediction Based on Sentimen News")
    st.markdown("### Prediksi Harga IHSG Berdasarkan Analisis Sentimen Berita Ekonomi")
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: Prediksi ini hanya untuk keperluan informasi dan analisis. 
    Bukan merupakan rekomendasi investasi. Keputusan investasi menjadi 
    tanggung jawab masing-masing investor. Pastikan untuk melakukan riset 
    dan konsultasi dengan ahli sebelum mengambil keputusan investasi.
    """)

    # About Section
    st.markdown("""
    ## 🎯 Tentang Aplikasi
        
    Aplikasi ini memprediksi pergerakan **Indeks Harga Saham Gabungan (IHSG)** dengan menggabungkan:
    - 🗞️ **Analisis Sentimen Berita** - Menganalisis berita ekonomi dan pasar saham
    - 🗂️ **Data Historis IHSG** - Pola pergerakan harga saham
    - ⚙️ **Machine Learning** - Model SARIMAX untuk prediksi akurat
        
    ### ✨ Fitur Utama
        
    1. **Prediksi Harian** - Prediksi harga penutupan IHSG besok
    2. **Sentimen Real-time** - Analisis sentimen dari berita terkini
    3. **Visualisasi Tren** - Grafik pergerakan IHSG mingguan
    4. **Update Otomatis** - Data diperbarui setiap hari
    """)
    st.markdown("---")
    
    # Methodology Section
    st.markdown("""
    ## 🔬 Metodologi
    
    ### Model SARIMAX
    
    **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)** 
    adalah model time series yang powerful untuk prediksi data berurutan.
    
    #### Proses Training
    
    1. **Data Collection** - Scraping berita dari sumber terpercaya
    2. **Sentiment Analysis** - Klasifikasi sentimen (Positif/Netral/Negatif)
    3. **Feature Engineering** - Membuat fitur dari harga & sentimen
    4. **Model Training** - Training SARIMAX dengan data historis
    5. **Evaluation** - Validasi dengan metrik MAPE, RMSE, MAE
    
    ### Sentimen Berita
    
    Sentimen diklasifikasikan menggunakan model **IndoBERT** yang sudah di-fine-tune 
    untuk bahasa Indonesia, dengan keyword yang telah disesuaikan dengan istilah finansial spesifik.
    """)
    st.markdown("---")
    
    # How to Read
    st.markdown("""
    ## 👀 Cara Membaca Dashboard
    
    ### Skor Sentimen
    - 🟢 **Positif** - Berita optimis, indikasi bullish (harga IHSG cenderung naik)
    - 🟡 **Netral** - Berita seimbang, tidak ada sinyal jelas
    - 🔴 **Negatif** - Berita pesimis, indikasi bearish (harga IHSG cenderung turun)
    
    ### Chart
    - **Line Solid** - Data aktual IHSG (6 hari terakhir)
    - **Line Continues** - Prediksi untuk besok (sambungan dari data aktual)
    - **Warna Pink** - Actual data
    - **Warna Biru** - Prediction (titik terakhir)
    
    ### Tips Penggunaan
    
    1. Lihat **Rata-rata sentimen** sebagai alat ukur untuk kondisi pasar secara umum
    2. Perhatikan **distribusi sentimen** - apakah mayoritas positif/negatif?
    3. Cek **grafik tren** untuk melihat momentum pergerakan
    4. Gunakan prediksi sebagai **salah satu referensi**, bukan satu-satunya dasar keputusan
    """)
    st.markdown("---")
    
    st.info('💡 **Ready?** Klik [**Dashboard**](#top_tabs) untuk melihat prediksi hari ini!')

# TAB 2: DASHBOARD
with tab2:
    # Load data
    df_news = load_today_news()
    df_pred = load_prediction()
    df_weekly = load_weekly_data()
    df_today = load_today_ihsg()
    
    # disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: Prediksi ini hanya untuk keperluan informasi dan analisis. 
    Bukan merupakan rekomendasi investasi. Keputusan investasi menjadi 
    tanggung jawab masing-masing investor. Pastikan untuk melakukan riset 
    dan konsultasi dengan ahli sebelum mengambil keputusan investasi.
    """)

    # Title
    st.markdown("<h1 style='text-align: center;'>IHSG Prediction based on news sentimen</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    val_today = df_today.iloc[0]['close'] if not df_today.empty else None

    with col1:
        if not df_today.empty:
            today_date = df_today.iloc[0]['date'].strftime('%Y/%m/%d')
            today_close = df_today.iloc[0]['close']
            st.markdown(f"""
            <div class='metric-label'>IHSG hari ini</div>
            <div class='metric-date'>({today_date})</div>
            """, unsafe_allow_html=True)
            st.markdown(f"<div class='big1-metric'>{today_close:,.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-label'>IHSG hari ini</div>
            <div class='metric-date'>(-)</div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='big1-metric'>-</div>", unsafe_allow_html=True)
    
    with col2:
        if not df_pred.empty and val_today:
            pred_date = df_pred.iloc[0]['date'].strftime('%Y/%m/%d')
            pred_close = df_pred.iloc[0]['predicted_close']
            
            diff = pred_close - val_today
            pct_change = (diff / val_today) * 100
            
            if diff > 0:
                color = "#27ae60"
                icon = "▲"
                sign = "+"
            elif diff < 0:
                color = "#e74c3c"
                icon = "▼"
                sign = ""
            else:
                color = "#888"
                icon = "▬"
                sign = ""
            
            st.markdown(f"""
            <div class='metric-label'>Prediksi IHSG Besok</div>
            <div class='metric-date'>({pred_date})</div>""", unsafe_allow_html=True)

            st.markdown(f"<div class='big2-metric'>{pred_close:,.2f}</div>", unsafe_allow_html=True)            
            st.markdown(f"""
            <div style='text-align: center; margin-top: -15px; font-weight: bold; color: {color}; font-size: 1.2rem;'>
                {icon} {sign}{pct_change:.2f}% ({sign}{diff:,.2f})
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='big2-metric'>-</div>", unsafe_allow_html=True)
    
    # Model info
    st.markdown("""
    <div class='info-box'>
        <p style='margin: 0; font-size: 0.9rem; color: #555;'>
            💡Prediksi dilakukan menggunakan model <strong>SARIMAX</strong>.
        </p>
    </div>""", unsafe_allow_html=True)
    
    # Main layout
    col_left, col_right = st.columns([3, 3])
    
    # LEFT: CHART
    with col_left:
        st.subheader("Grafik harga IHSG selama seminggu")
        
        if not df_weekly.empty and len(df_weekly) >= 2:
            # Create continuous line
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_weekly['date'],
                y=df_weekly['close'],
                mode='lines+markers',
                name='IHSG',
                line=dict(color="#00D4FF", width=3),
                marker=dict(
                    size=8,
                    color=['#00D4FF'] * (len(df_weekly) - 1) + ["#FF0088"],
                ),
                hovertemplate='%{x|%Y-%m-%d}<br>Close: %{y:,.2f}<extra></extra>'
            ))
            
            df_pred_segment = df_weekly.tail(2)
            fig.add_trace(go.Scatter(
                x=df_pred_segment['date'],
                y=df_pred_segment['close'],
                mode='lines',
                name='Prediksi',
                line=dict(color="#FF0088", width=3, dash='dash'),
                showlegend=True,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(
                    title="",
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                ),
                yaxis=dict(
                    title="",
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#A0A0A0')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data grafik tidak tersedia")
    
    # RIGHT: SENTIMENT + NEWS
    with col_right:
        # Sentiment
        st.subheader("📌 Ringkasan Sentimen Harian")
        
        if not df_news.empty:
            df_news['sentiment_label'] = df_news['sentiment_label'].str.lower()
            
            avg_sent = df_news["sentiment_score"].mean()
            dist = df_news["sentiment_label"].value_counts(normalize=True) * 100
            
            st.metric("Average Sentimen", f"{avg_sent:.3f}")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("🟢 Positif", f"{dist.get('positif', dist.get('positive', 0)):.0f}%")
            with col_s2:
                st.metric("🟡 Netral", f"{dist.get('netral', dist.get('neutral', 0)):.0f}%")
            with col_s3:
                st.metric("🔴 Negatif", f"{dist.get('negatif', dist.get('negative', 0)):.0f}%")
        else:
            st.info("Belum ada data sentimen hari ini")
        
        st.divider()
        
        # News
        col1_header, col2_header = st.columns([3, 1])
        with col1_header:
            st.subheader("Berita Ekonomi")
        with col2_header:
            if st.button("🔄 update berita"):
                num_to_show = min(5, len(df_news))
                st.session_state.news_idx = np.random.choice(df_news.index, num_to_show, replace=False)
                st.rerun()
        
        if not df_news.empty:
            news_date = df_news.iloc[0]['tanggal']
            if isinstance(news_date, str):
                news_date = pd.to_datetime(news_date).date()
            
            if "news_idx" not in st.session_state:
                num_to_show = min(5, len(df_news))
                st.session_state.news_idx = np.random.choice(df_news.index, num_to_show, replace=False)
            
            if news_date == TODAY.date():
                st.caption(f"📅 Berita hari ini ({news_date.strftime('%Y/%m/%d')})")
            else:
                st.caption(f"📅 Berita kemarin ({news_date.strftime('%Y/%m/%d')}) - Berita hari ini belum tersedia")
            
            for i, idx in enumerate(st.session_state.news_idx, 1):
                row = df_news.loc[idx]
                label = row["sentiment_label"].lower()
                
                if label in ["positif", "positive"]:
                    badge_color = "#27ae60"
                    badge_text = "sentiment"
                elif label in ["negatif", "negative"]:
                    badge_color = "#e74c3c"
                    badge_text = "sentiment"
                else:
                    badge_color = "#f39c12"
                    badge_text = "sentiment"
                
                url = row.get('url', '#')
                st.markdown(f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.05); 
                    padding: 1.2rem; 
                    border-radius: 12px; 
                    margin-bottom: 1.3rem; 
                    border: 1px solid rgba(128, 128, 128, 0.3);
                    border-left: 6px solid {badge_color};
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    transition: transform 0.5s ease;">
                    <div style="
                        font-size: 1.1rem; 
                        font-weight: 700; 
                        margin-bottom: 0.6rem;
                        line-height: 1.4;">
                        {row['judul']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                        <span style="
                            padding: 0.2rem 0.8rem; 
                            border-radius: 20px; 
                            background: {badge_color}; 
                            color: white; 
                            font-size: 0.7rem; 
                            font-weight: bold;
                            text-transform: uppercase;
                            letter-spacing: 0.05em;">
                            {label}
                        </span>
                        <a href='{url}' target='_blank' style='font-size: 0.8rem; color: #FF0088; text-decoration: none; font-weight: 600;'>
                            Baca berita selengkapnya!
                        </a>
                    </div>
                </div>""", unsafe_allow_html=True)
            
        else:
            st.info("Belum ada berita tersedia")
