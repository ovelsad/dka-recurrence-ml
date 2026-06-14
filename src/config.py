"""Единый источник путей, констант и настроек воспроизводимости.

Все модули и ноутбуки берут пути и seed отсюда, чтобы не дублировать строки
и гарантировать одинаковое поведение во всех экспериментах.
"""

from pathlib import Path

# Фиксированное зерно генератора случайных чисел.
# Используем везде: train/test split, кросс-валидация, модели, SMOTE.
RANDOM_SEED = 42

# Корень проекта вычисляем от расположения этого файла (src/config.py -> корень).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Каталоги с данными.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Каталоги с результатами.
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Сохраненные модели и пайплайны для онлайн-сервиса.
MODELS_DIR = PROJECT_ROOT / "models"

# Исходный датасет от врачей.
RAW_DATASET = RAW_DIR / "dka_bd_289.xlsx"

# Целевая переменная (факт рецидива): 0 - единичный эпизод, 1 - рецидив.
# В сырых данных 273 заполнено, 16 пропусков (строки без метки в обучении не
# используем как размеченные).
TARGET_COLUMN = "Рецидив (0 - единичный, 1 - рецидив)"

# Порог числа наблюдений для выбора критерия нормальности по протоколу статистика.
# Менее 50 - критерий Шапиро-Уилка, более 50 - критерий Колмогорова-Смирнова.
NORMALITY_N_THRESHOLD = 50

# Уровень значимости для статистических выводов.
ALPHA = 0.05

# Разрешение для экспорта графиков в статью.
FIGURE_DPI = 300


def ensure_dirs() -> None:
    """Создает каталоги для промежуточных данных и результатов, если их нет."""
    for path in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
