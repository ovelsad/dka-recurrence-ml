"""Клинические пороги признаков и их сочетаний (шаг 10.5)

Разведочный статистический анализ на всех пациентах с исходом, как шаг 1.
Для признака порог выбирается по максимуму индекса Юдена (чувствительность +
специфичность - 1). Направление берется по данным: если выше значение - чаще
рецидив, правило "не ниже порога", иначе "не выше порога". Порог и метрики
при нем найдены на тех же пациентах и поэтому оптимистичны, насколько порог
устойчив, показывает бутстрэп.

Сочетание двух признаков: оба условия выполнены против остальных, среди
пациентов, у которых известны оба признака. Доли рецидива, ОШ и p по
протоколу (stats.compare_categorical). Список PAIRS ограничен тремя
сочетаниями из статьи. Это разведочный анализ, а не зарегистрированные
до просмотра данных гипотезы
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, stats
from .eda import short_name
from .evaluate import youden_threshold

N_BOOTSTRAP = 1000  # ресэмплов для ДИ порога
DOSE = "Суточная доза инсулина"
RETINOPATHY = "Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)"
# ядро и второй ряд факторов шага 10
FEATURES = [DOSE, "HbA1c", "ЛПВП", config.DURATION, "ХБП, С", RETINOPATHY]
# сочетания с первым фактором всех моделей, суточной дозой
PAIRS = [(DOSE, "HbA1c"), (DOSE, "ЛПВП"), (DOSE, "ХБП, С")]


def direction(x: pd.Series, y: pd.Series) -> int:
    """1, если большее значение признака чаще у рецидива, иначе -1"""
    return 1 if roc_auc_score(y, x) >= 0.5 else -1


def cutoff(x: pd.Series, y: pd.Series, sign: int) -> float:
    """Порог по максимуму индекса Юдена в единицах признака"""
    return sign * youden_threshold(y, sign * x)


def meets(x: pd.Series, threshold: float, sign: int) -> pd.Series:
    """Выполнено ли условие: не ниже порога при sign 1, не выше при sign -1"""
    return x >= threshold if sign > 0 else x <= threshold


def rule_text(feature: str, threshold: float, sign: int) -> str:
    """Условие словами: "HbA1c ≤ 9.2" """
    value = f"{threshold:.2f}".rstrip("0").rstrip(".")
    return f"{short_name(feature)} {'≥' if sign > 0 else '≤'} {value}"


def bootstrap_cutoff(x: pd.Series, y: pd.Series, sign: int) -> tuple[float, float]:
    """95% ДИ порога бутстрэпом пациентов, направление фиксировано"""
    rng = np.random.default_rng(config.RANDOM_SEED)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y.iloc[idx].nunique() < 2:
            continue
        values.append(cutoff(x.iloc[idx], y.iloc[idx], sign))
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def accuracy(condition: pd.Series, y: pd.Series) -> dict:
    """Чувствительность и специфичность условия как правила для рецидива"""
    return {
        "чувствительность": (condition & (y == 1)).sum() / (y == 1).sum(),
        "специфичность": (~condition & (y == 0)).sum() / (y == 0).sum(),
    }


def comparison(condition: pd.Series, y: pd.Series) -> dict:
    """Доли рецидива при выполненном условии и остальных, ОШ и p по протоколу"""
    result = stats.compare_categorical(condition.astype(int), y)
    yes, no = y[condition], y[~condition]
    return {
        "пациентов с условием": int(condition.sum()),
        "рецидив при условии": f"{int(yes.sum())} из {len(yes)}",
        "доля при условии": yes.mean(),
        "рецидив без условия": f"{int(no.sum())} из {len(no)}",
        "доля без условия": no.mean(),
        "ОШ": result["or"],
        "ОШ ДИ низ": result["or_lo"],
        "ОШ ДИ верх": result["or_hi"],
        "критерий": result["test"],
        "p": result["p"],
    }


def threshold_row(feature: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Порог одного признака, его ДИ, чувствительность, специфичность и доли"""
    known = X[feature].notna()
    x, labels = X.loc[known, feature].astype(float), y[known]
    sign = direction(x, labels)
    threshold = cutoff(x, labels, sign)
    low, high = bootstrap_cutoff(x, labels, sign)
    condition = meets(x, threshold, sign)
    return {
        "признак": feature,
        "известно": int(known.sum()),
        "условие": rule_text(feature, threshold, sign),
        "направление": sign,
        "порог": threshold,
        "порог ДИ низ": low,
        "порог ДИ верх": high,
        **accuracy(condition, labels),
        **comparison(condition, labels),
    }


def threshold_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Пороги признаков FEATURES по индексу Юдена"""
    return pd.DataFrame([threshold_row(f, X, y) for f in FEATURES])


def pair_table(X: pd.DataFrame, y: pd.Series, single: pd.DataFrame) -> pd.DataFrame:
    """Сочетания PAIRS: оба условия с порогами из single против остальных"""
    rules = single.set_index("признак")
    rows = []
    for first, second in PAIRS:
        known = X[first].notna() & X[second].notna()
        both = pd.Series(True, index=X.index[known])
        for feature in (first, second):
            rule = rules.loc[feature]
            x = X.loc[known, feature].astype(float)
            both &= meets(x, rule["порог"], rule["направление"])
        rows.append(
            {
                "сочетание": f"{rules.loc[first, 'условие']} и {rules.loc[second, 'условие']}",
                "известны оба": int(known.sum()),
                **accuracy(both, y[known]),
                **comparison(both, y[known]),
            }
        )
    return pd.DataFrame(rows)
