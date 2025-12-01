import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import os

def load_any_data(path):
    """JSON veya CSV dosyasını otomatik okur ve fiyat sütununu döndürür."""

    ext = os.path.splitext(path)[1].lower()

    # JSON dosyası
    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data["archive"])

        # Altın fiyatı
        if "close_try" in df.columns:
            prices = df["close_try"].values.reshape(-1, 1)

        # USD fiyatı
        elif "close_usd" in df.columns:
            prices = df["close_usd"].values.reshape(-1, 1)
        else:
            raise ValueError("JSON dosyasında 'close_try' veya 'close_usd' bulunamadı.")

        return prices, df

    # CSV dosyası
    elif ext == ".csv":
        df = pd.read_csv(path)

        # USD close price CSV
        if "close" in df.columns:
            prices = df["close"].values.reshape(-1, 1)
            return prices, df
        else:
            raise ValueError("CSV içinde 'close' kolonu bulunamadı.")

    else:
        raise ValueError("Desteklenmeyen dosya formatı.")

def scale_data(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)
    return scaled, scaler

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

