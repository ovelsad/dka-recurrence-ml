"""Разведочный анализ (шаг 0): словарь колонок, мультиколлинеарность, графики

Функции возвращают таблицы и рисунки, сохранение и вывод оставляем ноутбуку.
Цвета групп проверены на различимость при нарушениях цветового зрения:
единичный эпизод синий, рецидив оранжевый
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .data import labeled
from .stats import level_key, level_label, level_name

GROUP_COLORS = {0: "#2a78d6", 1: "#eb6834"}
GROUP_NAMES = {0: "единичный эпизод", 1: "рецидив"}
INK = "#52514e"  # подписи и оси
GRID = "#e4e3df"
# расходящаяся шкала для корреляций: синий - серый ноль - красный
DIVERGING = LinearSegmentedColormap.from_list("corr", ["#2a78d6", "#f0efec", "#e34948"])

HISTOGRAM_BINS = 15
MAX_LISTED_LEVELS = 10  # в словаре колонок уровни перечисляем, если их не больше
LABEL_GAP = 0.02  # отступ подписи n над столбцом, в долях оси
FRIEDEWALD_DIVISOR = 2.2  # ТГ / 2,2 в ммоль/л (5 в мг/дл)
FRIEDEWALD_TOLERANCE = 0.1  # ммоль/л, расхождение в пределах округления


def short_name(column: str) -> str:
    """Короткое имя для подписей: без расшифровки кодов в скобках и без 'при поступлении'"""
    name = re.sub(r"\s*\(.*?\)", "", column)
    return name.replace(" при поступлении", "").strip()


def role(column: str) -> str:
    """Роль колонки базы в проекте: идентификатор, целевая, тип признака или исключен"""
    if column in (config.ID_COLUMN, config.CASE_ID):
        return "идентификатор"
    if column == config.TARGET:
        return "целевая"
    if column in config.FEATURES:
        return config.feature_type(column)
    return "исключен"


def column_dictionary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Словарь колонок базы по размеченным пациентам

    Роль, заполненность, уровни или диапазон значений, причина исключения
    """
    known = labeled(cleaned)
    rows = []
    for column in raw.columns:
        if column not in cleaned.columns:
            rows.append({"колонка": column, "роль": "пустая, удалена"})
            continue
        values = known[column]
        kind = role(column)
        if kind == "количественный":
            summary = f"{values.min():g} – {values.max():g}"
        elif values.nunique() <= MAX_LISTED_LEVELS:
            summary = ", ".join(
                level_label(v) for v in sorted(values.dropna().unique(), key=level_key)
            )
        else:
            summary = f"{values.nunique()} разных"
        rows.append(
            {
                "колонка": column,
                "роль": kind,
                "заполнено": int(values.notna().sum()),
                "пропуск %": round(values.isna().mean() * 100, 1),
                "значения": summary,
                "причина исключения": config.EXCLUDED.get(column, ""),
            }
        )
    return pd.DataFrame(rows)


def vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Фактор инфляции дисперсии 1 / (1 - R²) по строкам, где заполнены все колонки

    R² - доля дисперсии признака, объясненная линейной регрессией на остальные
    признаки набора. VIF выше 10 говорит, что признак почти линейно выражается
    через остальные. Считаем по небольшим клиническим блокам: на всех признаках
    разом полных строк почти не остается, а попарные корреляции по разным
    подвыборкам дают несогласованную матрицу и отрицательные VIF
    """
    complete = df[columns].dropna()
    if len(complete) <= len(columns):
        raise ValueError(
            f"Полных строк {len(complete)}, для VIF нужно больше {len(columns)}"
        )
    rows = []
    for column in columns:
        others = [complete[c] for c in columns if c != column]
        design = np.column_stack([np.ones(len(complete)), *others])
        coefficients, *_ = np.linalg.lstsq(design, complete[column], rcond=None)
        residual = complete[column] - design @ coefficients
        r2 = (
            1
            - (residual**2).sum()
            / ((complete[column] - complete[column].mean()) ** 2).sum()
        )
        value = np.inf if r2 >= 1 else 1 / (1 - r2)  # полная линейная зависимость
        rows.append(
            {"признак": short_name(column), "VIF": round(value, 1), "n": len(complete)}
        )
    return pd.DataFrame(rows)


def friedewald_check(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проверка, что ЛПНП рассчитан по Фридвальду: ОХ = ЛПНП + ЛПВП + ТГ / 2,2

    Возвращает две таблицы. Первая сравнивает коэффициенты регрессии ОХ на три
    липида с коэффициентами формулы и дает R² для регрессии и для самой формулы.
    Вторая - пациенты, у которых формула не выполняется в пределах допуска
    """
    lipids = [config.LDL, config.HDL, config.TRIGLYCERIDES]
    complete = df[[config.ID_COLUMN, config.TOTAL_CHOLESTEROL, *lipids]].dropna()
    total = complete[config.TOTAL_CHOLESTEROL]
    spread = ((total - total.mean()) ** 2).sum()

    design = np.column_stack([np.ones(len(complete)), *(complete[c] for c in lipids)])
    coefficients, *_ = np.linalg.lstsq(design, total, rcond=None)
    regression_r2 = 1 - ((total - design @ coefficients) ** 2).sum() / spread

    formula = (
        complete[config.LDL]
        + complete[config.HDL]
        + complete[config.TRIGLYCERIDES] / FRIEDEWALD_DIVISOR
    )
    difference = total - formula
    formula_r2 = 1 - (difference**2).sum() / spread
    matched = difference.abs() <= FRIEDEWALD_TOLERANCE

    summary = pd.DataFrame(
        {
            "член": ["свободный", config.LDL, config.HDL, config.TRIGLYCERIDES, "R²"],
            "формула Фридвальда": [0.0, 1.0, 1.0, 1 / FRIEDEWALD_DIVISOR, formula_r2],
            "регрессия на всех": [*coefficients, regression_r2],
        }
    ).round(3)
    summary.attrs = {"n": len(complete), "совпало": int(matched.sum())}
    mismatches = complete.loc[
        ~matched, [config.ID_COLUMN, config.TOTAL_CHOLESTEROL]
    ].assign(
        **{
            "по формуле": formula[~matched].round(2),
            "разница": difference[~matched].round(2),
        }
    )
    return summary, mismatches.sort_values("разница")


def excluded_pairs(df: pd.DataFrame) -> dict:
    """Исключенные признаки против выражения через оставшиеся

    Возраст манифестации против возраста минус длительность СД, ЛПНП против
    ОХ - ЛПВП - ТГ / 2,2 (формула Фридвальда). Возвращает {исключенный признак:
    таблица пар без пропусков}
    """
    age = df["Возраст (на текущий момент)"] - df[config.DURATION]
    friedewald = (
        df[config.TOTAL_CHOLESTEROL]
        - df[config.HDL]
        - df[config.TRIGLYCERIDES] / FRIEDEWALD_DIVISOR
    )
    return {
        config.MANIFESTATION: pd.DataFrame(
            {"исключенный": df[config.MANIFESTATION], "выражение": age}
        ).dropna(),
        config.LDL: pd.DataFrame(
            {"исключенный": df[config.LDL], "выражение": friedewald}
        ).dropna(),
    }


# ---------------------------------------------------------------- графики


def plot_excluded(pairs: dict) -> plt.Figure:
    """Рассеяние исключенного признака против его выражения через оставшиеся

    В заголовке R² линейной регрессии и число пациентов с известными значениями
    """
    labels = {
        config.MANIFESTATION: "возраст - длительность СД, лет",
        config.LDL: "ОХ - ЛПВП - ТГ / 2,2, ммоль/л",
    }
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.4))
    for ax, (column, table) in zip(np.atleast_1d(axes), pairs.items()):
        r2 = np.corrcoef(table["выражение"], table["исключенный"])[0, 1] ** 2
        ax.scatter(
            table["выражение"],
            table["исключенный"],
            s=12,
            color=GROUP_COLORS[0],
            alpha=0.6,
        )
        low = min(table.min())
        high = max(table.max())
        ax.plot([low, high], [low, high], color="gray", lw=1, ls="--")
        title = f"{short_name(column)}: R² = {r2:.3f}, n = {len(table)}"
        ax.set(
            xlabel=labels[column],
            ylabel=short_name(column),
            title=title.replace(".", ","),
        )
        style(ax)
    fig.tight_layout()
    return fig


def add_caption(fig, text: str) -> None:
    """Подпись под рисунком: что показано и как читать"""
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=9, color=INK, wrap=True)


def annotate_cells(ax, texts: list, dark: np.ndarray) -> None:
    """Подписи в ячейках тепловой карты

    texts[i][j] - текст ячейки строки i и столбца j, dark - где фон темный,
    там текст белый
    """
    for i, row in enumerate(texts):
        for j, text in enumerate(row):
            color = "white" if dark[i, j] else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)


def grid(n: int, ncols: int = 4, cell: tuple = (3.6, 2.8)):
    """Сетка осей под n графиков, лишние оси скрыты"""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def style(ax) -> None:
    """Сдержанные оси: без верхней и правой рамки, светлая сетка по y"""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def group_legend(fig) -> None:
    """Общая легенда групп над сеткой графиков"""
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in (0, 1)]
    fig.legend(
        handles,
        [GROUP_NAMES[g] for g in (0, 1)],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )


def plot_distributions(df: pd.DataFrame, columns: list):
    """Гистограммы количественных признаков по группам исхода

    Высота - плотность, чтобы группы разного размера сравнивались по форме
    """
    known = labeled(df)
    fig, axes = grid(len(columns))
    for ax, column in zip(axes, columns):
        values = known[column].dropna()
        bins = np.histogram_bin_edges(values, bins=HISTOGRAM_BINS)
        for group in (0, 1):
            sample = known.loc[known[config.TARGET] == group, column].dropna()
            ax.hist(
                sample,
                bins=bins,
                density=True,
                histtype="step",
                lw=2,
                color=GROUP_COLORS[group],
            )
        ax.set_title(f"{short_name(column)} (n={len(values)})", fontsize=9, color=INK)
        ax.set_yticks([])
        style(ax)
    group_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_boxplots(df: pd.DataFrame, columns: list):
    """Ящики с усами количественных признаков по группам исхода (усы 1,5 IQR)"""
    known = labeled(df)
    fig, axes = grid(len(columns))
    for ax, column in zip(axes, columns):
        samples = [
            known.loc[known[config.TARGET] == g, column].dropna() for g in (0, 1)
        ]
        parts = ax.boxplot(
            samples,
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "#0b0b0b", "lw": 1.5},
            flierprops={"marker": "o", "ms": 3, "mec": INK, "alpha": 0.6},
        )
        for patch, group in zip(parts["boxes"], (0, 1)):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_edgecolor(GROUP_COLORS[group])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2], [f"n={len(s)}" for s in samples])
        ax.set_title(short_name(column), fontsize=9, color=INK)
        style(ax)
    group_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_category_rates(df: pd.DataFrame, columns: list):
    """Доля рецидива по уровням категориальных признаков

    Над столбцом число пациентов уровня, пропуск показан отдельным уровнем.
    Пунктир - доля рецидива во всей размеченной выборке
    """
    known = labeled(df)
    overall = known[config.TARGET].mean()
    fig, axes = grid(len(columns), ncols=3, cell=(4.2, 2.8))
    for ax, column in zip(axes, columns):
        rates = known.groupby(known[column].map(level_label))[config.TARGET].agg(
            ["mean", "size"]
        )
        rates = rates.loc[sorted(rates.index, key=level_key)]
        positions = np.arange(len(rates))
        ax.bar(positions, rates["mean"], width=0.6, color=GROUP_COLORS[1])
        for x, (rate, size) in zip(positions, rates.itertuples(index=False)):
            ax.text(
                x, rate + LABEL_GAP, f"n={size}", ha="center", fontsize=7, color=INK
            )
        ax.axhline(overall, color=INK, lw=1, ls="--")
        names = [level_name(column, label) for label in rates.index]
        ax.set_xticks(positions, names, fontsize=8, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(short_name(column), fontsize=9, color=INK)
        style(ax)
    fig.suptitle(
        f"Доля рецидива по уровням признака, пунктир - {overall:.0%} во всей выборке",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_correlation(df: pd.DataFrame, columns: list):
    """Тепловая карта корреляций Спирмена с подписями значений"""
    corr = df[columns].corr(method="spearman")
    labels = [short_name(c) for c in columns]
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                f"{corr.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#0b0b0b",
            )
    ax.set_xticks(range(len(columns)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), labels, fontsize=8)
    ax.set_title("Корреляции Спирмена", fontsize=11, color=INK)
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig


def plot_missing(df: pd.DataFrame, columns: list):
    """Доля пропусков по признакам модели на размеченных пациентах"""
    share = (labeled(df)[columns].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [short_name(c) for c in share.index], share, color=GROUP_COLORS[0], height=0.6
    )
    for y, value in enumerate(share):
        ax.text(value + 1, y, f"{value:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("пропуск, %", color=INK)
    ax.tick_params(axis="y", labelsize=8)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    return fig
