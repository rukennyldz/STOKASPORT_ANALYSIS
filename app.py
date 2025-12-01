import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os # YENİ EKLENDİ
from models.arima_model import run_arima

st.set_page_config(page_title="StokaSport Dashboard", layout="wide")

# --- SESSION STATE FIX ---
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
# --------------------------

# ======================================================
# 1) VERİ YÜKLEME FONKSİYONU (GLOBAL - HATASIZ SON VERSİYON)
# ======================================================
def process_json_file(path):
    """
    JSON veya CSV dosyasını okur, DataFrame'e çevirir ve zaman serisi (pd.Series) haline getirir.
    'update_date' ve esnek fiyat sütunlarını destekler.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        df = pd.DataFrame()
        
        # --- Veri Yükleme ---
        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            
            # 'archive' anahtarını veya doğrudan listeyi destekle
            if isinstance(data, dict) and "archive" in data:
                df = pd.DataFrame(data["archive"])
            else:
                df = pd.DataFrame(data)

        elif ext == ".csv":
            df = pd.read_csv(path)
        
        else:
            raise ValueError("Desteklenmeyen dosya formatı.")
        
        if df.empty:
             raise ValueError("Yüklenen dosya boş veya okunamadı.")

        # --- Tarih İşleme ('update_date' dahil) ---
        date_col = None
        for col in ["Date", "date", "Time", "update_date"]: 
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            if date_col == "update_date":
                # Unix Timestamp (saniye) -> Tarih dönüşümü
                df["Date_Index"] = pd.to_datetime(df[date_col], unit='s', errors='coerce') 
            else:
                df["Date_Index"] = pd.to_datetime(df[date_col], errors='coerce')
                
            df = df.set_index("Date_Index").sort_index()
            
            # 🔥 HATA DÜZELTME: KeyError yerine index'teki NaT değerlerini filtrele
            df = df[df.index.notna()] 
        else:
            raise ValueError("Uygun tarih sütunu bulunamadı: ('Date', 'date', 'Time' veya 'update_date').")


        # --- Kapanış Fiyatı Sütunu Seçimi ---
        price_col = None
        for col in ["close_try", "close_usd", "close", "Close"]:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            series = df[price_col].astype(float)
        else:
            raise ValueError(f"Dosyada uygun fiyat sütunu ('close_try', 'close_usd', 'close' veya 'Close') bulunamadı.")

        series = series.fillna(method="ffill")
        return series.rename("Close")

    except FileNotFoundError:
        st.error(f"❌ Veri dosyası bulunamadı: {path}")
        return pd.Series(dtype="float64")

    except Exception as e:
        # Daha anlaşılır bir hata mesajı için
        st.error(f"❌ Veri işleme hatası ({path}): {e}")
        return pd.Series(dtype="float64")

# ======================================================
# 2) VERİLERİ YÜKLE (GLOBAL)
# ======================================================
gold_series = process_json_file("data/1-gram-altin.json")
usd_series  = process_json_file("data/1-USD.json")

# ======================================================
# 3) SIDEBAR AYARLARI
# ======================================================
st.sidebar.header("StokaSport Ayarları")

asset = st.sidebar.selectbox(
    "Varlık Seç:",
    ("Gram Altın (XAU/TRY)", "USD/TRY")
)

st.sidebar.subheader("Tahmin Ayarları")
prediction_days = st.sidebar.slider(
    "Tahmin Periyodu (Gün)",
    7, 30, 14
)

if st.sidebar.button("ARIMA Tahminini Çalıştır"):
    st.session_state.show_prediction = True

# ======================================================
# 4) SEÇİLEN VARLIĞA GÖRE VERİ
# ======================================================
if asset == "Gram Altın (XAU/TRY)":
    series = gold_series.copy()
    title = "Gram Altın Fiyatları"
else:
    series = usd_series.copy()
    title = "USD/TRY Fiyatları"

if series.empty:
    # process_json_file'da hata oluştuysa veya dosya bulunamadıysa burada durur.
    st.error("Veri serisi yüklenemedi. Lütfen dosya yollarınızı ve formatınızı kontrol edin.")
    st.stop()

# ======================================================
# 5) GRAFİKLER VE ARIMA TAHMİNİ
# ======================================================
st.title("📊 StokaSport Finansal Analiz ve Tahmin Uygulaması")
st.subheader(f"{title} - Tarihsel Veri ve Tahmin")

forecast_series = pd.Series(dtype="float64")

# ARIMA ÇALIŞTIR
if st.session_state.show_prediction:
    st.info(f"{asset} için ARIMA(5,1,0) ile {prediction_days} günlük tahmin yapılıyor...")

    try:
        # ARIMA için seriyi kopyalamak, modelin orjinal veriyi değiştirmesini engeller (iyi pratik)
        forecast_series = run_arima(series.copy(), steps=prediction_days)
        st.session_state.forecast_data = forecast_series
    except Exception as e:
        st.error(f"ARIMA Model Hatası: {e}")
        st.session_state.show_prediction = False

# ======================================================
# 6) PLOTLY GRAFİĞİ
# ======================================================
fig = go.Figure()

# Gerçek fiyat
fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    mode="lines",
    name="Gerçekleşen Fiyat",
    line=dict(color="blue")
))

# Tahmin
if st.session_state.show_prediction and not forecast_series.empty:
    # Gerçek veri serisinin son noktasını alarak tahmin çizgisi ile birleştirme
    last_real_date = series.index[-1]
    last_real_price = series.values[-1]

    forecast_dates = [last_real_date] + list(forecast_series.index)
    forecast_prices = [last_real_price] + list(forecast_series.values)

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode="lines",
        name=f"ARIMA Tahmini ({prediction_days} Gün)",
        line=dict(color="red", dash="dash")
    ))

fig.update_layout(
    title=f"{title} Zaman Serisi Analizi",
    yaxis_title="Fiyat (TRY)",
    xaxis_title="Tarih"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 7) TAHMİN TABLOSU
# ======================================================
if st.session_state.show_prediction and not forecast_series.empty:
    st.subheader("📅 Tahmin Edilen Fiyatlar")

    df_forecast = pd.DataFrame(forecast_series)
    df_forecast.index.name = "Tarih"
    df_forecast.columns = ["Tahmini Fiyat"]

    # Tahmin serisinin index'ini tarihe dönüştürme (zaten pandas.DatetimeIndex)
    st.dataframe(df_forecast.style.format({"Tahmini Fiyat": "{:.2f}"}))

# ======================================================
# 8) RETURN & VOLATILITE METRİKLERİ
# ======================================================
# Hesaplamaları sadece veri boş değilse yap
if not series.empty:
    df = series.to_frame(name="Close")
    df["Return"] = df["Close"].pct_change()

    # Getiriyi ve volatiliteyi yıllık bazda hesapla (252 işlem günü varsayımıyla)
    annual_return = df["Return"].mean() * 252
    volatility = df["Return"].std() * np.sqrt(252)

    st.subheader("⚙️ Tarihsel Performans Metrikleri")
    left, right = st.columns(2)

    with left:
        st.metric("Yıllık Ortalama Getiri", f"{annual_return:.2%}")

    with right:
        st.metric("Yıllık Volatilite", f"{volatility:.2%}")

# ======================================================
# 9) RAW DATA
# ======================================================
with st.expander("Son Veri Seti"):
    st.write(df.tail(30).style.format({"Close": "{:.4f}"}))