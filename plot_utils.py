import numpy as np
import matplotlib.pyplot as plt

def lstm_plot(actual, train_predict, test_predict, look_back, train_len, save=False):
    """Eğitim ve test tahminlerini grafikte gösterir"""

    train_plot = np.empty_like(actual)
    train_plot[:] = np.nan
    train_plot[look_back:train_len] = train_predict

    test_plot = np.empty_like(actual)
    test_plot[:] = np.nan
    test_plot[train_len:train_len + len(test_predict)] = test_predict

    plt.figure(figsize=(15,7))
    plt.plot(actual, color='green', label='Gerçek Fiyat')
    plt.plot(train_plot, color='red', label='Eğitim Tahmini')
    plt.plot(test_plot, color='blue', label='Test Tahmini')
    plt.title('Gram Altın LSTM Fiyat Tahmini')
    plt.xlabel('Gün İndeksi')
    plt.ylabel('TRY')
    plt.legend()

    if save:
        plt.savefig("lstm_gold_forecast.png")

    return plt
