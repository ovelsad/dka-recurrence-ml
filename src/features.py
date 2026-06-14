"""Подготовка к обучению: разбиение, наборы признаков, отбор Boruta (этап 6).

Train/test split делаем один раз и стратифицированно по рецидиву. Обучаемся
только на размеченных пациентах (16 строк без метки откладываем). Наборы
признаков задают ось перебора. Boruta - алгоритмический отбор на train.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import columns, config
from .config import RANDOM_SEED

TARGET = config.TARGET_COLUMN
TEST_SIZE = 0.2

# Признак, выводимый из возраста и длительности (мультиколлинеарность).
COLLINEAR_DROP = "Возраст манифестации СД"

# Значимые по Таблице 1 после поправки Бенджамини-Хохберга.
SIGNIFICANT = [
    "Длительность СД (лет)",
    "Суточная доза инсулина",
    "HbA1c",
    "ЛПВП",
    "Вид инсулинотерапии (1 - ручки, 2 - помпа) на момент ДКА",
    "ХБП, С",
    "ХБП, А",
    "Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)",
    "Невролог",
    "Алкоголь за сутки до ДКА (0 - нет, 1 - да)",
]


def all_features(df: pd.DataFrame) -> list:
    """Все признаки без идентификаторов и целевой."""
    roles = columns.classify(df)
    return [c for c in roles["quantitative"] + roles["categorical"] if c in df.columns]


def split_xy(df: pd.DataFrame):
    """Размеченная часть: X (все признаки) и y. Строки без метки отброшены."""
    labeled = df[df[TARGET].notna()].copy()
    y = labeled[TARGET].astype(int)
    X = labeled[all_features(df)]
    return X, y


def make_split(df: pd.DataFrame):
    """Стратифицированное разбиение train/test, детерминированное по seed."""
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test


def feature_sets(df: pd.DataFrame) -> dict:
    """Наборы признаков для перебора (без Boruta, он считается на train)."""
    feats = all_features(df)
    return {
        "all": feats,
        "no_collinear": [c for c in feats if c != COLLINEAR_DROP],
        "significant": [c for c in SIGNIFICANT if c in df.columns],
    }


def column_types(df: pd.DataFrame, feats: list):
    """Делит набор признаков на количественные и категориальные."""
    roles = columns.classify(df)
    quant = [c for c in feats if c in roles["quantitative"]]
    cat = [c for c in feats if c in roles["categorical"]]
    return quant, cat


if __name__ == "__main__":
    from . import io
    df = io.load_processed()
    X_train, X_test, y_train, y_test = make_split(df)
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"баланс train: {dict(y_train.value_counts())}")
    sets = feature_sets(df)
    for name, feats in sets.items():
        print(f"набор {name}: {len(feats)} признаков")
