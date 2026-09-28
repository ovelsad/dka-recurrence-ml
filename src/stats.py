"""Статистика шага 1 строго по протоколу врачей (StatTech 4.11.2)

Две группы (рецидив против единичного эпизода):
- количественные: нормальность (Шапиро-Уилк при n < 50, иначе Лиллиефорс),
  равенство дисперсий по Левену, затем t-критерий Стьюдента, U-критерий
  Манна-Уитни или W-критерий Бруннера-Мюнцеля
- категориальные: хи-квадрат Пирсона при ожидаемом не ниже 10, иначе точный
  критерий Фишера, для четырехпольных таблиц отношение шансов с 95% ДИ

Три группы (степень тяжести ДКА):
- количественные: ANOVA с апостериорным критерием Тьюки при нормальности и равных
  дисперсиях, иначе Краскел-Уоллис с апостериорным критерием Данна и поправкой
  Холма
- категориальные: хи-квадрат Пирсона, попарно хи-квадрат с поправкой Холма

Описание по всей выборке: M и SD с 95% ДИ среднего либо Me с квартилями,
категориальные - абсолютные значения, доли и 95% ДИ по Клопперу-Пирсону.

Значимость по сырому p < 0,05 без поправки на множественность. Значимый p
помечаем звездочкой. Числа печатаем с десятичной запятой, как в таблицах врачей
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests

from . import config

# Самый малый p, который печатаем числом, меньший пишем как < 0,001
SMALLEST_PRINTED_P = 0.001
# Минимум наблюдений в группе, при котором сравнение имеет смысл
MIN_GROUP_SIZE = 2
NO_DATA = "нет данных"

# Расшифровка кодов уровней словами, как их пишут врачи. Колонки без расшифровки
# (стадии ХБП, тип СД, степень тяжести) печатаем кодом, врачи делают так же
LEVEL_NAMES = {
    "Пол (0 - Ж, 1 - М)": {"0": "женский", "1": "мужской"},
    "Вид инсулинотерапии (1 - ручки, 2 - помпа) на момент ДКА": {
        "0": "не изв",
        "1": "МИИ",
        "2": "ППИИ",
    },
    "Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)": {
        "0": "нет",
        "1": "непролиферативная",
        "2": "препролиферативная",
        "3": "пролиферативная",
    },
    "Тяжелые гипогликемии в анамнезе (0 - нет, 1 - да)": {"0": "нет", "1": "да"},
    "Невролог": {"0": "нет", "1": "да"},
    "Алкоголь за сутки до ДКА (0 - нет, 1 - да)": {"0": "нет", "1": "да"},
}


# ---------------------------------------------------------------- уровни и формат


def level_label(value) -> str:
    """Уровень категориального строкой: 1.0 в '1', пропуск в 'пропуск'"""
    if pd.isna(value):
        return "пропуск"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def level_key(value) -> tuple[float, str]:
    """Ключ сортировки уровней: по числу или рангу стадии ХБП, текст и пропуск в конце

    Вторая часть ключа разводит уровни с одним рангом: '3' идет раньше '3a'
    """
    label = level_label(value)
    if label in config.CKD_RANK:
        return config.CKD_RANK[label], label
    try:
        return float(label), label
    except ValueError:  # текстовый уровень или 'пропуск'
        return np.inf, label


def level_name(column: str, value) -> str:
    """Уровень словами по расшифровке врачей, при ее отсутствии сам код"""
    label = level_label(value)
    return LEVEL_NAMES.get(column, {}).get(label, label)


def num(value, digits: int = 2) -> str:
    """Число строкой в русской записи, с десятичной запятой"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{digits}f}".replace(".", ",")


def p_text(p: float, digits: int = 3) -> str:
    """p без звездочки: очень малый как < 0,001, иначе = число"""
    if np.isnan(p):
        return "не определен"
    return "< 0,001" if p < SMALLEST_PRINTED_P else f"= {num(p, digits)}"


def format_p(p, digits: int = 3) -> str:
    """p для колонки таблицы: < 0,001 или число, значимый со звездочкой"""
    if p is None or np.isnan(p):
        return ""
    star = "*" if p < config.ALPHA else ""
    return p_text(p, digits).removeprefix("= ") + star


def format_or(result: dict) -> str:
    """ОШ с 95% ДИ строкой, пусто для многопольных таблиц"""
    if np.isnan(result["or"]):
        return ""
    return f"{num(result['or'])} ({num(result['or_lo'])}; {num(result['or_hi'])})"


def count_share(count: int, total: int) -> str:
    """Абсолютное значение и доля в процентах: 12 (34,56%)"""
    if total == 0:
        return f"{count} ({NO_DATA})"
    return f"{count} ({num(count / total * 100)}%)"


def quartiles(x) -> tuple[float, float, float]:
    """Нижний квартиль, медиана, верхний квартиль"""
    q1, median, q3 = np.percentile(np.asarray(x, dtype=float), [25, 50, 75])
    return q1, median, q3


def group_cell(x, normal: bool) -> str:
    """Значение количественного в группе: M (SD) или Me [Q1; Q3]"""
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return NO_DATA
    if normal:
        return f"{num(x.mean())} ({num(x.std(ddof=1))})"
    q1, median, q3 = quartiles(x)
    return f"{num(median)} [{num(q1)}; {num(q3)}]"


# ---------------------------------------------------------------- базовые расчеты


def normality_pvalue(x) -> float:
    """p критерия нормальности по протоколу

    Шапиро-Уилк при n менее порога, иначе Лиллиефорс. При n менее 3 проверка
    невозможна, возвращаем NaN
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if x.size < 3:
        return np.nan
    if x.size < config.NORMALITY_N_THRESHOLD:
        return float(scipy_stats.shapiro(x).pvalue)
    return float(lilliefors(x, dist="norm")[1])


def is_normal(x) -> bool:
    """Распределение нормально, если p критерия не ниже alpha"""
    p = normality_pvalue(x)
    return bool(not np.isnan(p) and p >= config.ALPHA)


def mean_ci(x) -> tuple[float, float, float, float]:
    """Среднее, стандартное отклонение и 95% ДИ среднего по t-распределению"""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if x.size < 2:
        return (float(x[0]) if x.size else np.nan, np.nan, np.nan, np.nan)
    mean, sd = x.mean(), x.std(ddof=1)
    half = scipy_stats.t.ppf(0.975, x.size - 1) * sd / np.sqrt(x.size)
    return mean, sd, mean - half, mean + half


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """95% ДИ доли методом Клоппера-Пирсона (точный, на бета-распределении)"""
    low = 0.0 if k == 0 else float(scipy_stats.beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(scipy_stats.beta.ppf(0.975, k + 1, n - k))
    return low, high


def odds_ratio(a: float, b: float, c: float, d: float) -> tuple[float, float, float]:
    """Отношение шансов (a*d)/(b*c) и 95% ДИ логарифмическим методом Вульфа

    a, b - экспонированные с исходом и без, c, d - неэкспонированные с исходом и
    без. При нулевой клетке добавляем 0.5 ко всем (поправка Холдейна-Анскомба)
    """
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    value = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = scipy_stats.norm.ppf(0.975)
    low = float(np.exp(np.log(value) - z * se))
    high = float(np.exp(np.log(value) + z * se))
    return float(value), low, high


# ---------------------------------------------------------------- две группы


def compare_quantitative(x0, x1) -> dict:
    """Сравнение двух групп по количественному, критерий по протоколу

    Критерий выбираем по нормальности обеих групп и равенству дисперсий (Левен на
    медиане). Нормальность при неравных дисперсиях протокол не описывает, для нее
    берем t-критерий Уэлча и помечаем это в названии
    """
    x0 = pd.Series(x0).dropna().to_numpy(dtype=float)
    x1 = pd.Series(x1).dropna().to_numpy(dtype=float)
    if min(x0.size, x1.size) < MIN_GROUP_SIZE:
        return {"test": "не применимо (мало наблюдений)", "p": np.nan, "normal": False}

    normal = is_normal(x0) and is_normal(x1)
    equal_var = bool(scipy_stats.levene(x0, x1, center="median").pvalue >= config.ALPHA)
    if normal and equal_var:
        test, p = "t-критерий Стьюдента", scipy_stats.ttest_ind(x0, x1).pvalue
    elif normal:
        test = "t-критерий Уэлча (нормальность, неравные дисперсии, вне протокола)"
        p = scipy_stats.ttest_ind(x0, x1, equal_var=False).pvalue
    elif equal_var:
        test, p = "U-критерий Манна-Уитни", scipy_stats.mannwhitneyu(x0, x1).pvalue
    else:
        test, p = (
            "W-критерий Бруннера-Мюнцеля",
            scipy_stats.brunnermunzel(x0, x1).pvalue,
        )
    return {"test": test, "p": float(p), "normal": normal}


def compare_categorical(feature, outcome) -> dict:
    """Сравнение долей между двумя группами, критерий по протоколу

    Четырехпольная таблица: хи-квадрат Пирсона при минимальном ожидаемом не ниже
    10, иначе точный критерий Фишера, плюс отношение шансов. Многопольная:
    хи-квадрат Пирсона без единого ОШ
    """
    pairs = pd.DataFrame({"feature": pd.Series(feature), "outcome": pd.Series(outcome)})
    table = pd.crosstab(pairs["feature"], pairs["outcome"])
    result = {"test": None, "p": np.nan, "or": np.nan, "or_lo": np.nan, "or_hi": np.nan}
    if table.shape[1] < 2:
        result["test"] = "не применимо (пустая группа)"
        return result
    if table.shape[0] < 2:
        result["test"] = "не применимо (один уровень)"
        return result

    counts = table.to_numpy()
    if table.shape == (2, 2):
        # границу "ровно 10" относим к хи-квадрату
        if scipy_stats.contingency.expected_freq(counts).min() >= 10:
            result["test"] = "хи-квадрат Пирсона"
            result["p"] = float(
                scipy_stats.chi2_contingency(counts, correction=False)[1]
            )
        else:
            result["test"] = "точный критерий Фишера"
            result["p"] = float(scipy_stats.fisher_exact(counts)[1])
        # строки - уровни признака по возрастанию (0, 1), столбцы - исход (0, 1),
        # экспозиция это верхний уровень признака
        d, c = counts[0, 0], counts[0, 1]
        b, a = counts[1, 0], counts[1, 1]
        result["or"], result["or_lo"], result["or_hi"] = odds_ratio(a, b, c, d)
    else:
        result["test"] = "хи-квадрат Пирсона"
        result["p"] = float(scipy_stats.chi2_contingency(counts, correction=False)[1])
    return result


def outcome_from_masks(group0: pd.Series, group1: pd.Series) -> pd.Series:
    """Метка группы 0/1 по двум маскам строк, строки вне обеих масок - NaN"""
    if (group0 & group1).any():
        raise ValueError(
            "Маски групп пересекаются: строка не может быть в обеих группах"
        )
    outcome = pd.Series(np.nan, index=group0.index)
    outcome[group0], outcome[group1] = 0, 1
    return outcome


def two_groups_quantitative(
    df: pd.DataFrame, columns: list, group0: pd.Series, group1: pd.Series
) -> pd.DataFrame:
    """Таблица количественных по двум группам: значения, n, критерий и p

    Группы задаются масками строк, чтобы тем же кодом сравнивать и подгруппы
    """
    outcome_from_masks(group0, group1)  # проверка, что маски не пересекаются
    rows = []
    for column in columns:
        x0, x1 = df.loc[group0, column].dropna(), df.loc[group1, column].dropna()
        result = compare_quantitative(x0, x1)
        normal = result["normal"]
        rows.append(
            {
                "показатель": f"{column}, {'M (SD)' if normal else 'Me [IQR]'}",
                "нет": group_cell(x0, normal),
                "да": group_cell(x1, normal),
                "n нет": len(x0),
                "n да": len(x1),
                "критерий": result["test"],
                "p": format_p(result["p"]),
            }
        )
    return pd.DataFrame(rows)


def two_groups_categorical(
    df: pd.DataFrame, columns: list, group0: pd.Series, group1: pd.Series
) -> pd.DataFrame:
    """Таблица категориальных по двум группам: строка на уровень признака

    В строке абс. (%) внутри каждой группы, на первой строке признака критерий, p
    и ОШ. Долю считаем от непропущенных в группе, чтобы доли уровней давали 100%
    """
    outcome = outcome_from_masks(group0, group1)
    rows = []
    for column in columns:
        known = df[column].notna() & outcome.notna()
        values, groups = df.loc[known, column], outcome[known]
        result = compare_categorical(values, groups)
        total0, total1 = int((groups == 0).sum()), int((groups == 1).sum())
        for i, level in enumerate(sorted(values.unique(), key=level_key)):
            first = i == 0
            rows.append(
                {
                    "показатель": f"{column}, абс. (%)" if first else "",
                    "категория": level_name(column, level),
                    "нет": count_share(
                        int(((values == level) & (groups == 0)).sum()), total0
                    ),
                    "да": count_share(
                        int(((values == level) & (groups == 1)).sum()), total1
                    ),
                    "критерий": result["test"] if first else "",
                    "p": format_p(result["p"]) if first else "",
                    "ОШ (95% ДИ)": format_or(result) if first else "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- три группы


def group_pairs(n_groups: int) -> list:
    """Пары групп для апостериорных сравнений: (0, 1) это p1-2"""
    return list(itertools.combinations(range(n_groups), 2))


def format_pairs(pairs: list, pvalues: list) -> str:
    """Апостериорные p строкой: p1-2 = 0,012; p2-3 < 0,001"""
    return "; ".join(
        f"p{i + 1}-{j + 1} {p_text(p)}" for (i, j), p in zip(pairs, pvalues)
    )


def dunn_holm(samples: list) -> list:
    """Критерий Данна с поправкой на связанные ранги, p с поправкой Холма"""
    values = np.concatenate(samples)
    n = len(values)
    ranks = scipy_stats.rankdata(values)
    _, counts = np.unique(values, return_counts=True)
    ties = np.sum(counts**3 - counts) / (12 * (n - 1))
    bounds = np.cumsum([0] + [len(s) for s in samples])
    mean_rank = [ranks[bounds[k] : bounds[k + 1]].mean() for k in range(len(samples))]
    raw = []
    for i, j in group_pairs(len(samples)):
        se = np.sqrt(
            (n * (n + 1) / 12 - ties) * (1 / len(samples[i]) + 1 / len(samples[j]))
        )
        z = abs(mean_rank[i] - mean_rank[j]) / se
        raw.append(2 * scipy_stats.norm.sf(z))
    return list(multipletests(raw, method="holm")[1])


def compare_several_quantitative(samples: list) -> dict:
    """Сравнение трех и более групп по количественному, критерий по протоколу

    Нормальность при неравных дисперсиях протокол не описывает, для нее берем
    ANOVA Уэлча с критерием Геймса-Хауэлла и помечаем это в названии
    """
    if min(len(s) for s in samples) < MIN_GROUP_SIZE:
        return {
            "test": "не применимо (мало наблюдений)",
            "p": np.nan,
            "pairs": [],
            "normal": False,
        }

    normal = all(is_normal(s) for s in samples)
    equal_var = scipy_stats.levene(*samples, center="median").pvalue >= config.ALPHA
    pairs = group_pairs(len(samples))
    if normal and equal_var:
        test = "ANOVA, апостериорно Тьюки"
        p = scipy_stats.f_oneway(*samples).pvalue
        posthoc = scipy_stats.tukey_hsd(*samples).pvalue
        pvalues = [posthoc[i, j] for i, j in pairs]
    elif normal:
        test = "ANOVA Уэлча, апостериорно Геймс-Хауэлл (вне протокола)"
        p = scipy_stats.f_oneway(*samples, equal_var=False).pvalue
        posthoc = scipy_stats.tukey_hsd(*samples, equal_var=False).pvalue
        pvalues = [posthoc[i, j] for i, j in pairs]
    else:
        test = "Краскел-Уоллис, апостериорно Данн с поправкой Холма"
        p = scipy_stats.kruskal(*samples).pvalue
        pvalues = dunn_holm(samples)
    return {"test": test, "p": float(p), "pairs": pvalues, "normal": normal}


def chi2_pvalue(table: pd.DataFrame) -> float:
    """p хи-квадрата Пирсона по таблице без пустых строк и столбцов"""
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        return np.nan
    return float(scipy_stats.chi2_contingency(table.to_numpy(), correction=False)[1])


def pairwise_chi2_holm(table: pd.DataFrame) -> list:
    """Попарные хи-квадрат Пирсона между столбцами-группами, p с поправкой Холма"""
    raw = np.array(
        [chi2_pvalue(table.iloc[:, [i, j]]) for i, j in group_pairs(table.shape[1])]
    )
    adjusted = np.full(raw.size, np.nan)  # пара с пустой группой остается без p
    defined = ~np.isnan(raw)
    adjusted[defined] = multipletests(raw[defined], method="holm")[1]
    return list(adjusted)


def three_groups_quantitative(
    df: pd.DataFrame, columns: list, by: str, levels: list
) -> pd.DataFrame:
    """Таблица количественных по группам колонки by

    Общий p с 4 знаками, как в образце врачей, апостериорные p при значимом общем
    """
    pairs = group_pairs(len(levels))
    rows = []
    for column in columns:
        samples = [
            df.loc[df[by] == g, column].dropna().to_numpy(dtype=float) for g in levels
        ]
        result = compare_several_quantitative(samples)
        row = {
            "показатель": f"{column}, {'M (SD)' if result['normal'] else 'Me [IQR]'}"
        }
        for level, sample in zip(levels, samples):
            row[level_label(level)] = group_cell(sample, result["normal"])
        significant = result["p"] < config.ALPHA
        row["p"] = format_p(result["p"], digits=4)
        row["апостериорно"] = (
            format_pairs(pairs, result["pairs"]) if significant else ""
        )
        row["критерий"] = result["test"]
        rows.append(row)
    return pd.DataFrame(rows)


def three_groups_categorical(
    df: pd.DataFrame, columns: list, by: str, levels: list
) -> pd.DataFrame:
    """Таблица категориальных по группам колонки by: абс. (%) внутри группы

    Общий p - хи-квадрат Пирсона, при значимом общем попарные хи-квадрат с
    поправкой Холма. Минимальное ожидаемое выводим рядом: при значении меньше 5
    p ориентировочный
    """
    pairs = group_pairs(len(levels))
    rows = []
    for column in columns:
        known = df[df[column].notna()]
        table = pd.crosstab(known[column], known[by]).reindex(
            columns=levels, fill_value=0
        )
        table = table[table.sum(axis=1) > 0]
        p = chi2_pvalue(table)
        posthoc = (
            format_pairs(pairs, pairwise_chi2_holm(table)) if p < config.ALPHA else ""
        )
        nonempty = table.loc[:, table.sum(axis=0) > 0]
        expected_min = float(
            scipy_stats.contingency.expected_freq(nonempty.to_numpy()).min()
        )
        totals = table.sum(axis=0)
        for k, level in enumerate(sorted(table.index, key=level_key)):
            first = k == 0
            row = {
                "показатель": f"{column}, абс. (%)" if first else "",
                "категория": level_name(column, level),
            }
            for group in levels:
                row[level_label(group)] = count_share(
                    int(table.at[level, group]), int(totals[group])
                )
            row["p"] = format_p(p, digits=4) if first else ""
            row["апостериорно"] = posthoc if first else ""
            row["ожидаемое мин"] = num(expected_min) if first else ""
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- вся выборка


def describe_quantitative(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Описательная статистика количественных по всей выборке

    Печатаем обе формы рядом (M ± SD с 95% ДИ и Me с квартилями), метка у
    названия говорит, какая из них основная по критерию нормальности
    """
    rows = []
    for column in columns:
        x = df[column].dropna().astype(float)
        if x.empty:
            continue
        mean, sd, low, high = mean_ci(x)
        q1, median, q3 = quartiles(x)
        rows.append(
            {
                "показатель": f"{column}, {'M ± SD' if is_normal(x) else 'Me'}",
                "M ± SD / Me": f"{num(mean)} ± {num(sd)} / {num(median)}",
                "95% ДИ / Q1 – Q3": f"[{num(low)}; {num(high)}] / [{num(q1)}; {num(q3)}]",
                "n": len(x),
                "min": num(x.min()),
                "max": num(x.max()),
            }
        )
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Описательная статистика категориальных по всей выборке

    Абсолютные значения, доля от непропущенных и 95% ДИ доли по Клопперу-Пирсону
    """
    rows = []
    for column in columns:
        values = df[column].dropna()
        total = len(values)
        for i, level in enumerate(sorted(values.unique(), key=level_key)):
            count = int((values == level).sum())
            low, high = clopper_pearson(count, total)
            rows.append(
                {
                    "показатель": column if i == 0 else "",
                    "категория": level_name(column, level),
                    "абс.": count,
                    "%": num(count / total * 100),
                    "95% ДИ": f"{num(low * 100)} – {num(high * 100)}",
                }
            )
    return pd.DataFrame(rows)
