"""Загрузка и сохранение данных проекта."""

from pathlib import Path

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Читает исходный датасет из data/raw."""
    return pd.read_excel(config.RAW_DATASET)


def load_processed(name: str = "dka_clean") -> pd.DataFrame:
    """Читает очищенную таблицу из data/processed."""
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")


def save_interim(df: pd.DataFrame, name: str) -> Path:
    """Сохраняет промежуточную таблицу в data/interim в формате parquet."""
    config.ensure_dirs()
    path = config.INTERIM_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Сохраняет финальную таблицу под обучение в data/processed."""
    config.ensure_dirs()
    path = config.PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path
