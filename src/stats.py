"""Статистический протокол проекта. Канон, переданный медицинским статистиком.

Реализует выбор критериев и описание показателей строго по согласованным
правилам. Используется в EDA, в этапе статистических тестов и в skill /stat-test.

Нормальность: Шапиро-Уилка при n менее 50, Колмогорова-Смирнова (в варианте
Лиллиефорса, с оценкой параметров по выборке) при n не менее 50.
Описание количественных: при нормальности M, SD и 95% ДИ среднего; иначе Me и
квартили Q1-Q3. Категориальные: абсолютные значения, доли, 95% ДИ Клоппера-Пирсона.
Сравнение двух групп: Стьюдент (нормальность и равные дисперсии), Манна-Уитни
(не нормальное, равные дисперсии), Бруннера-Мюнцеля (не нормальное, неравные
дисперсии). Доли: хи-квадрат Пирсона при ожидаемом более 10, точный критерий
Фишера при ожидаемом менее 10 для четырехпольных таблиц, хи-квадрат для
многопольных. Мера эффекта: отношение шансов с 95% ДИ.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from .config import ALPHA, NORMALITY_N_THRESHOLD


@dataclass
class NormalityResult:
    is_normal: bool
    test: str
    statistic: float
    pvalue: float
    n: int


def _numeric(series) -> np.ndarray:
    """Превращает в массив чисел без пропусков."""
    x = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return x.to_numpy()


def check_normality(series, alpha: float = ALPHA) -> NormalityResult:
    """Проверка нормальности по протоколу: Шапиро-Уилка или Колмогорова-Смирнова."""
    x = _numeric(series)
    n = x.size
    if n < 3:
        return NormalityResult(False, "недостаточно данных", np.nan, np.nan, n)
    if n < NORMALITY_N_THRESHOLD:
        stat, p = stats.shapiro(x)
        name = "Шапиро-Уилка"
    else:
        # Лиллиефорс - критерий Колмогорова-Смирнова с оценкой параметров.
        stat, p = lilliefors(x, dist="norm")
        name = "Колмогорова-Смирнова (Лиллиефорса)"
    return NormalityResult(p > alpha, name, float(stat), float(p), n)


def describe_quantitative(series, alpha: float = ALPHA) -> dict:
    """Описание количественного показателя по протоколу."""
    raw = pd.Series(series)
    x = _numeric(raw)
    n = x.size
    missing = int(raw.isna().sum())
    norm = check_normality(x, alpha)

    result = {
        "n": n,
        "missing": missing,
        "normal": norm.is_normal,
        "normality_test": norm.test,
        "normality_p": norm.pvalue,
    }
    if n == 0:
        result["summary"] = "нет данных"
        return result

    if norm.is_normal:
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        if n > 1:
            sem = sd / np.sqrt(n)
            lo, hi = stats.t.interval(1 - alpha, n - 1, loc=mean, scale=sem)
        else:
            lo, hi = np.nan, np.nan
        result.update({"mean": mean, "sd": sd, "ci_low": lo, "ci_high": hi})
        result["summary"] = (
            f"M={mean:.2f}, SD={sd:.2f}, 95% ДИ [{lo:.2f}; {hi:.2f}]")
    else:
        me = float(np.median(x))
        q1 = float(np.quantile(x, 0.25))
        q3 = float(np.quantile(x, 0.75))
        result.update({"median": me, "q1": q1, "q3": q3})
        result["summary"] = f"Me={me:.2f}, Q1-Q3 [{q1:.2f}; {q3:.2f}]"
    return result


def describe_categorical(series, alpha: float = ALPHA) -> pd.DataFrame:
    """Описание категориального показателя: n, доля, 95% ДИ Клоппера-Пирсона."""
    s = pd.Series(series)
    n_valid = int(s.notna().sum())
    counts = s.value_counts(dropna=True)

    rows = []
    for value, count in counts.items():
        prop = count / n_valid if n_valid else np.nan
        lo, hi = proportion_confint(count, n_valid, alpha=alpha, method="beta")
        rows.append({
            "категория": value,
            "n": int(count),
            "доля_%": round(prop * 100, 1),
            "ДИ_низ_%": round(lo * 100, 1),
            "ДИ_верх_%": round(hi * 100, 1),
        })
    table = pd.DataFrame(rows)
    table.attrs["n_valid"] = n_valid
    table.attrs["missing"] = int(s.isna().sum())
    return table


@dataclass
class ComparisonResult:
    test: str
    statistic: float
    pvalue: float
    note: str = ""


def compare_two_groups(values, groups, alpha: float = ALPHA) -> ComparisonResult:
    """Сравнение двух групп по количественному показателю по протоколу."""
    df = pd.DataFrame({"v": pd.to_numeric(pd.Series(values), errors="coerce"),
                       "g": pd.Series(groups).values})
    df = df.dropna()
    levels = sorted(df["g"].unique())
    if len(levels) != 2:
        return ComparisonResult("неприменимо", np.nan, np.nan,
                                "нужно ровно две группы")

    a = df.loc[df["g"] == levels[0], "v"].to_numpy()
    b = df.loc[df["g"] == levels[1], "v"].to_numpy()
    if a.size < 3 or b.size < 3:
        return ComparisonResult("недостаточно данных", np.nan, np.nan, "")

    normal = check_normality(a, alpha).is_normal and check_normality(b, alpha).is_normal
    equal_var = stats.levene(a, b, center="median").pvalue > alpha

    if normal and equal_var:
        stat, p = stats.ttest_ind(a, b, equal_var=True)
        return ComparisonResult("t-критерий Стьюдента", float(stat), float(p))
    if not normal and equal_var:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return ComparisonResult("U-критерий Манна-Уитни", float(stat), float(p))
    if not normal and not equal_var:
        stat, p = stats.brunnermunzel(a, b, alternative="two-sided")
        return ComparisonResult("W-критерий Бруннера-Мюнцеля", float(stat), float(p))
    # Нормальное распределение при неравных дисперсиях протокол не описывает.
    # Применяем t-критерий Уэлча, помечаем для согласования со статистиком.
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return ComparisonResult("t-критерий Уэлча", float(stat), float(p),
                            "случай вне протокола: нормальность при неравных дисперсиях")


def compare_proportions(table) -> ComparisonResult:
    """Сравнение долей: хи-квадрат Пирсона или точный критерий Фишера."""
    table = np.asarray(table)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    is_2x2 = table.shape == (2, 2)
    if is_2x2 and expected.min() < 10:
        _, p_fisher = stats.fisher_exact(table)
        return ComparisonResult("точный критерий Фишера", np.nan, float(p_fisher),
                                f"минимальное ожидаемое {expected.min():.1f}")
    return ComparisonResult("хи-квадрат Пирсона", float(chi2), float(p),
                            f"минимальное ожидаемое {expected.min():.1f}")


def adjust_pvalues(pvalues, method: str = "fdr_bh") -> np.ndarray:
    """Поправка на множественные сравнения. По умолчанию Бенджамини-Хохберга.

    Пропуски (NaN) не участвуют в поправке, на их месте возвращается NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    mask = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return adjusted
    adjusted[mask] = multipletests(p[mask], method=method)[1]
    return adjusted


def odds_ratio(table, alpha: float = ALPHA) -> dict:
    """Отношение шансов с 95% ДИ для четырехпольной таблицы."""
    t = Table2x2(np.asarray(table, dtype=float))
    lo, hi = t.oddsratio_confint(alpha)
    return {"odds_ratio": float(t.oddsratio), "ci_low": float(lo),
            "ci_high": float(hi)}
