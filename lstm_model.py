import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    """LSTM model mimarisini oluşturur"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_lstm(model, X_train, y_train, epochs=50, batch_size=1):
    """LSTM modelini eğitir"""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def predict_lstm(model, X):
    """Model ile tahmin üretir"""
    return model.predict(X)
