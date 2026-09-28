"""Модели сервиса до открытия теста: запись зафиксированных моделей (шаг 11)

Запись в data/interim/final_models.json делается до открытия теста. По каждой
модели сервиса: параметры модели, обученной на всем train, пороги и отпечаток
данных. У логрега еще коэффициенты, у CatBoost контрольная сумма его прогнозов
на train (коэффициентов у него нет). На каждом прогоне модели обучаются заново
тем же кодом, и запись сверяется с сохраненной: так видно, что после открытия
теста ничего не менялось.

Запись не перезаписывается молча. Если данные или модели изменились, freeze
падает. Тогда старую запись переименовываем в final_models_<дата>.json и
замораживаем заново осознанно. Прежняя запись одного логрега лежит в
finalist_2026-09-26.json
"""

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .tuning import describe, fingerprint

FROZEN_PATH = config.INTERIM_DIR / "final_models.json"
DIGITS = 6  # знаков у коэффициентов и прогнозов: сверка не зависит от шума округления


def parameters(fitted) -> dict:
    """Все параметры пайплайна в стабильном текстовом виде"""
    return {
        key: describe(value)
        for key, value in sorted(fitted.get_params(deep=True).items())
    }


def logreg_record(fitted) -> dict:
    """Параметры и коэффициенты логрега, обученного на всем train"""
    names = list(fitted[:-1].get_feature_names_out())
    model = fitted.named_steps["model"]
    return {
        "параметры": parameters(fitted),
        "коэффициенты": {
            name: round(float(c), DIGITS) for name, c in zip(names, model.coef_[0])
        },
        "свободный член": round(float(model.intercept_[0]), DIGITS),
    }


def prediction_checksum(fitted, X: pd.DataFrame) -> str:
    """Контрольная сумма прогнозов модели на X, округленных до DIGITS знаков"""
    predicted = np.round(fitted.predict_proba(X)[:, 1], DIGITS)
    return hashlib.sha256(predicted.tobytes()).hexdigest()[:16]


def model_record(name: str, fitted, X: pd.DataFrame) -> dict:
    """Запись одной модели: у логрега коэффициенты, у деревьев контрольная сумма"""
    if name.startswith("логрег"):
        return logreg_record(fitted)
    return {
        "параметры": parameters(fitted),
        "прогнозы train": prediction_checksum(fitted, X),
    }


def service_record(
    fitted_models: dict, X: pd.DataFrame, y: pd.Series, thresholds: dict
) -> dict:
    """Запись моделей сервиса для заморозки

    fitted_models - {название: модель, обученная на X, y}, thresholds -
    {название: {правило: порог}}
    """
    return {
        "дата": datetime.now().astimezone().date().isoformat(),
        "данные": fingerprint(X, y),
        "модели": {
            name: {
                **model_record(name, fitted, X),
                "пороги": {
                    rule: round(float(t), DIGITS)
                    for rule, t in thresholds[name].items()
                },
            }
            for name, fitted in fitted_models.items()
        },
    }


def freeze(record: dict) -> dict:
    """Сохраняет запись моделей или сверяет ее с уже сохраненной

    Дата в сверку не входит: повторный прогон ноутбука на тех же данных дает ту
    же запись и возвращает сохраненную, с датой первой заморозки
    """
    current = json.loads(json.dumps(record))  # вид после записи в json
    if FROZEN_PATH.exists():
        saved = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        changed = sorted(
            key for key in current if key != "дата" and saved.get(key) != current[key]
        )
        if changed:
            raise RuntimeError(
                f"Модели уже заморожены {saved['дата']}, изменилось: {changed}. "
                "Старую запись переименовать и заморозить заново осознанно"
            )
        return saved
    FROZEN_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return current
