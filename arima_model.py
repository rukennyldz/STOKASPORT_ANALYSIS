import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima(series, steps=7):
    """
    ARIMA modelini çalıştırır ve belirtilen adım sayısı kadar ileriye dönük tahmin yapar.

    Args:
        series (pd.Series): Tahmin edilecek zaman serisi (Kapanış fiyatları).
        steps (int): Kaç adım ileri tahmin yapılacağı.

    Returns:
        pd.Series: Tahmin edilen değerler (index'i tahmin başlangıç tarihinden itibaren devam eder).
    """
    try:
        model = ARIMA(series, order=(5,1,0))
        fitted = model.fit()
        
        start_date = series.index[-1] + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=steps, freq='D') 

        forecast_results = fitted.predict(start=len(series), end=len(series) + steps - 1)
        
        forecast_series = pd.Series(forecast_results.values, index=future_dates, name="ARIMA Tahmini")
        
        return forecast_series
    
    except Exception as e:
        # Streamlit uygulamasında kullanıcıya hata göstermek için bu print yerine 
        # st.error(f"ARIMA Tahmin Hatası: {e}") kullanabilirsiniz.
        print(f"ARIMA hatası: {e}")
        return pd.Series()