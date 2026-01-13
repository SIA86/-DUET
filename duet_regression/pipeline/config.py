
from dataclasses import dataclass
from pydantic import Field



@dataclass
class DUETConfig:
    # =========================
    # Общие параметры
    # =========================
    timestamp_col: str = "timestamp" # Название колонки с таймстэмпом
    features: list[str] = Field(default_factory=list)
    forecast: list[str] = Field(default_factory=list)
    not_to_normalise: list[str] = Field(default_factory=list)
    scaler: str = 'STD'             # Тип нормализации (STD, MINMAX, QUANT)
    seq_len: int = 96               # Длина входной последовательности
    horizon: int = 24               # Длина выходного прогноза
    patch_len: int = 16             # Длина патча (TCM)
    stride: int = 8                 # Шаг между патчами (TCM)
    moving_avg: int = 25            # Размер окна скользящего сглаживания

    # =========================
    # Параметры модели
    # =========================
    d_model: int = 64               # Размерность скрытого пространства в attention
    d_ff: int = 128                 # Размерность feedforward слоя
    n_heads: int = 4                # Количество голов в multi-head attention
    e_layers: int = 2               # Количество слоев в encoder (CCM)
    dropout: float = 0.1            # Dropout во всех слоях attention
    fc_dropout: float = 0.1         # Dropout в выходном head слое
    activation: str = "gelu"        # Активационная функция (relu, gelu, elu)
    num_experts: int = 4            # Число экспертов (в Router, если используется)

    # =========================
    # Режимы обработки
    # =========================
    CI: bool = True                 # Channel-Independent режим (если False — shared weights)
    use_router: bool = False        # Включить распределительный роутер
    timeenc: int = 1                # Использовать time encoding (0 = без, 1 = sin/cos и т.п.)

    # =========================
    # Настройки обучения
    # =========================
    batch_size: int = 32            # Размер батча
    epochs: int = 50                # Количество эпох
    learning_rate: float = 0.001    # Скорость обучения
    weight_decay: float = 1e-5      # L2 регуляризация
    patience: int = 5               # Патенс для early stopping
    loss: str = "mse"               # Функция потерь: mse, mae, smape, mase
    metric: str = "mae"             # Основная метрика: mae, smape, mase

    # =========================
    # Прочее
    # =========================
    seed: int = 42                  # Фиксированное зерно генератора случайных чисел
    verbose: bool = True            # Печать хода обучения
    checkpoint_best: str = ''    # Путь для сохранения лучших по val_accuracy весов
    checkpoint_final: str = ''     # Путь для сохранения финальных весов
