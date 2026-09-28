"""Загрузка исправленной базы, приведение типов, матрица признаков для моделей

Врачи сами исправили базу: убрали дубли, эпизоды рецидивистов и ошибки ввода.
Поэтому разбор текста здесь минимальный: пустая строка или пробел означают
пропуск, остальное должно читаться как число. Если врачи пришлют новую версию
с опечаткой, код остановится с ошибкой, а не заполнит значение молча
"""

import numpy as np
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Читает исправленную базу с рабочего листа"""
    return pd.read_excel(config.RAW_DATASET, sheet_name=config.RAW_SHEET)


def to_number(series: pd.Series) -> pd.Series:
    """Колонка в числа: пустая строка и пробелы становятся пропуском

    Любой другой текст вызывает ошибку с названием колонки
    """
    blanks_as_na = series.mask(series.astype(str).str.strip() == "")
    try:
        return pd.to_numeric(blanks_as_na).astype(float)
    except ValueError as error:
        raise ValueError(f"{series.name!r}: нечисловое значение ({error})") from error


def ckd_stage(df: pd.DataFrame) -> pd.Series:
    """Стадия ХБП по С строкой ('0'...'5', '3a', '3b'), неизвестная стадия - ошибка"""
    stage = df[config.CKD_STAGE].map(
        lambda v: np.nan if pd.isna(v) else str(v).strip().lower()
    )
    unknown = stage.notna() & ~stage.isin(config.CKD_RANK.keys())
    if unknown.any():
        rows = df.loc[unknown, config.ID_COLUMN].tolist()
        raise ValueError(
            f"Неизвестная стадия ХБП у N пп {rows}: {stage[unknown].tolist()}"
        )
    return stage


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы колонок, нужных статистике и моделям, остальное не трогает

    Количественные и категориальные коды становятся числами, стадия ХБП по С -
    строкой. Колонки, пустые на 100%, убираем
    """
    if config.TARGET not in raw.columns:
        raise ValueError(f"В базе нет целевой колонки {config.TARGET!r}")
    df = raw.drop(columns=[c for c in raw.columns if raw[c].notna().sum() == 0])

    numeric = config.STATS_QUANTITATIVE + config.STATS_CATEGORICAL + [config.TARGET]
    for column in numeric:
        if column != config.CKD_STAGE:
            df[column] = to_number(df[column])
    df[config.CKD_STAGE] = ckd_stage(df)
    return df


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Строки с известным исходом, исход целым числом"""
    known = df[df[config.TARGET].notna()].copy()
    known[config.TARGET] = known[config.TARGET].astype(int)
    return known


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Матрица признаков модели и исход

    Все признаки числовые: стадия ХБП по С переводится в ранг по config.CKD_RANK,
    остальные категориальные остаются кодами. Номинальные кодируются OHE позже,
    внутри пайплайна модели. Пропуски не заполняем. Индекс строк - N пп, по нему
    закреплен состав теста
    """
    known = labeled(df).set_index(config.ID_COLUMN)
    features = known[config.FEATURES].copy()
    features[config.CKD_STAGE] = known[config.CKD_STAGE].map(config.CKD_RANK)
    return features.astype(float), known[config.TARGET]
