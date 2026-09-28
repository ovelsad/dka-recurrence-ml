"""Одномерная разделяющая сила признаков на train (описание к шагу 7)

По каждому признаку модели считаем, насколько он один отделяет рецидив от
единичного эпизода:
- ROC-AUC, база 0,5
- PR-AUC (average precision), база это доля рецидивов на train

Обе метрики с бутстрэп-ДИ, ориентир силы - нижняя граница ДИ, а не точка.
Порядковые и бинарные признаки идут числом кода, количественные как есть. У
номинальных порядка нет, метрики для них не считаем. Каждый признак оцениваем
на своих непропущенных строках, поэтому подвыборки у признаков разные.

Таблица описательная: отбор признаков для модели делается на шаге 6 внутри
фолдов, иначе оценка качества завысится. Считаем только на train
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .eda import INK, short_name
from .evaluate import bootstrap_auc

METRICS = [
    "ROC-AUC",
    "ROC-AUC ДИ низ",
    "ROC-AUC ДИ верх",
    "PR-AUC",
    "PR-AUC ДИ низ",
    "PR-AUC ДИ верх",
]
COLUMNS = ["признак", "тип", *METRICS, "пропуск %"]


def oriented(feature: pd.Series, outcome: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Непропущенные пары значение-исход, значение развернуто так, что выше - выше риск

    Разворот (для протективного признака с ROC-AUC ниже 0,5) делаем один раз по
    точечной оценке, чтобы бутстрэп-интервал не смещался вверх
    """
    known = feature.notna().to_numpy()
    scores = feature.to_numpy(dtype=float)[known]
    labels = outcome.to_numpy(dtype=int)[known]
    if np.unique(labels).size == 2 and roc_auc_score(labels, scores) < 0.5:
        scores = -scores
    return scores, labels


def separation(feature: pd.Series, outcome: pd.Series) -> dict:
    """ROC-AUC и PR-AUC признака с бутстрэп-ДИ, NaN при одном классе"""
    scores, labels = oriented(feature, outcome)
    if np.unique(labels).size < 2:
        return {}
    roc_lo, roc_hi, pr_lo, pr_hi = bootstrap_auc(labels, scores)
    return {
        "ROC-AUC": roc_auc_score(labels, scores),
        "ROC-AUC ДИ низ": roc_lo,
        "ROC-AUC ДИ верх": roc_hi,
        "PR-AUC": average_precision_score(labels, scores),
        "PR-AUC ДИ низ": pr_lo,
        "PR-AUC ДИ верх": pr_hi,
    }


def screen(X: pd.DataFrame, y: pd.Series, excluded: list | None = None) -> pd.DataFrame:
    """Таблица разделяющей силы признаков train по убыванию ROC-AUC

    X - матрица признаков из data.model_frame. excluded - столбцы X, исключенные
    из модели: они идут с типом "исключен", чтобы было видно, могли ли они
    попасть в топ. Для номинальных метрики пустые
    """
    excluded = excluded or []
    rows = []
    for feature in X.columns:
        kind = "исключен" if feature in excluded else config.feature_type(feature)
        row = {"признак": feature, "тип": kind}
        if kind != "номинальный":
            row.update(separation(X[feature], y))
        row["пропуск %"] = X[feature].isna().mean() * 100
        rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    return table.sort_values("ROC-AUC", ascending=False, na_position="last").round(3)


def plot_forest(table: pd.DataFrame):
    """Forest-plot ROC-AUC с ДИ по признакам, где AUC определен, возвращает Figure"""
    shown = table[table["ROC-AUC"].notna()].sort_values("ROC-AUC")
    positions = range(len(shown))
    left = (shown["ROC-AUC"] - shown["ROC-AUC ДИ низ"]).clip(lower=0)
    right = (shown["ROC-AUC ДИ верх"] - shown["ROC-AUC"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.errorbar(
        shown["ROC-AUC"],
        positions,
        xerr=[left, right],
        fmt="o",
        color="#2a78d6",
        capsize=4,
    )
    ax.axvline(0.5, color="gray", lw=1, ls="--", label="случайное угадывание")
    ax.set_yticks(list(positions))
    labels = [
        short_name(name) + (" (исключен)" if kind == "исключен" else "")
        for name, kind in zip(shown["признак"], shown["тип"])
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("одномерный ROC-AUC (ориентированный)")
    ax.set_title("Разделяющая сила признаков на train с 95% ДИ (бутстрэп)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_selection_frequency(frequency: pd.DataFrame) -> plt.Figure:
    """Тепловая карта: доля фолдов, в которых признак отобран, по вариантам в столбцах"""
    row_height = 0.32  # дюйма на строку, чтобы подписи признаков не налезали
    fig, ax = plt.subplots(figsize=(6, row_height * len(frequency) + 1.5))
    image = ax.imshow(frequency.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(frequency.shape[0]):
        for j in range(frequency.shape[1]):
            value = frequency.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.6 else INK,  # выше 0,6 фон уже темный
            )
    ax.set_xticks(range(frequency.shape[1]), frequency.columns)
    ax.set_yticks(range(frequency.shape[0]), frequency.index, fontsize=8)
    ax.set_title("Частота отбора признака по фолдам CV")
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig
