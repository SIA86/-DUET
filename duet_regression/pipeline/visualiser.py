import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast_dual_axis(df: pd.DataFrame,
                             price_col="Close",
                             target_col="target",
                             forecast_col="forecast",
                             start: int = 0,
                             length: int = 200,
                             title: str = "DUET Forecast vs Target"):
    """
    Визуализирует прогноз модели:
    - Верхний график: цена (Close)
    - Нижний график: target и forecast

    Аргументы:
        df: DataFrame с данными
        price_col: имя колонки с ценой
        target_col: имя таргета
        forecast_col: имя предсказаний
        start: начальный индекс
        length: количество точек
        title: заголовок графика
    """
    end = start + length
    df_plot = df.iloc[start:end]

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Верхний график — цена
    axs[0].plot(df_plot[price_col], color='black', label='Close Price')
    axs[0].set_ylabel("Price")
    axs[0].legend()
    axs[0].grid(True)

    # Нижний график — target vs forecast
    axs[1].plot(df_plot[target_col], label='Target', color='blue', alpha=0.7)
    axs[1].plot(df_plot[forecast_col], label='Forecast', color='orange', linestyle='--')
    axs[1].set_ylabel("Value")
    axs[1].set_xlabel("Time")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
