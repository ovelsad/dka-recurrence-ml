"""Поколоночный разведочный анализ по протоколу проекта.

Для каждого количественного показателя: описание (M/SD/CI или Me/Q1-Q3),
проверка нормальности, сравнение по группам целевой переменной. Для каждого
категориального: доли с 95% ДИ Клоппера-Пирсона, таблица сопряженности с целевой,
выбор хи-квадрат или Фишера, отношение шансов для бинарных.

Также строит корреляции и VIF для оценки мультиколлинеарности. Графики сохраняет
в reports/figures, текстовый отчет в reports/eda_columns.md.
"""

import re

import matplotlib
matplotlib.use("Agg")  # неинтерактивный бэкенд для запуска скриптом
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import columns, config, io, stats


def setup_style() -> None:
    """Единый стиль графиков. Шрифт DejaVu Sans поддерживает кириллицу."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = config.FIGURE_DPI


def _safe_name(col: str) -> str:
    """Делает из имени колонки безопасное имя файла."""
    name = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", col).strip("_")
    return name[:60]


def eda_quantitative(df: pd.DataFrame, col: str, target: str) -> dict:
    """Описание количественной колонки и сравнение по группам целевой."""
    desc = stats.describe_quantitative(df[col])

    comparison = None
    if target and target in df.columns:
        sub = df[[col, target]].dropna()
        if sub[target].nunique() == 2:
            comparison = stats.compare_two_groups(sub[col], sub[target])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = pd.to_numeric(df[col], errors="coerce")
    sns.histplot(x.dropna(), kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Распределение: {col}", fontsize=9)
    axes[0].set_xlabel("")
    if target and target in df.columns:
        sns.boxplot(data=df, x=target, y=x, ax=axes[1], hue=df[target],
                    palette="Set2", legend=False)
        sns.stripplot(data=df, x=target, y=x, ax=axes[1], color=".3", size=3,
                      alpha=0.5)
        axes[1].set_title("По группам рецидива", fontsize=9)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"q_{_safe_name(col)}.png"
    fig.savefig(path)
    plt.close(fig)

    return {"описание": desc.get("summary"),
            "нормальность": f"{desc.get('normality_test')}, p={desc.get('normality_p'):.4f}"
            if not np.isnan(desc.get("normality_p", np.nan)) else desc.get("normality_test"),
            "пропуски": desc.get("missing"),
            "сравнение с целевой": (f"{comparison.test}, p={comparison.pvalue:.4f}"
                                    if comparison else "нет"),
            "фигура": path.name}


def eda_categorical(df: pd.DataFrame, col: str, target: str) -> dict:
    """Описание категориальной колонки и связь с целевой."""
    table = stats.describe_categorical(df[col])

    assoc = None
    odds = None
    if target and target in df.columns:
        ct = pd.crosstab(df[col], df[target])
        if ct.shape[0] >= 2 and ct.shape[1] == 2:
            assoc = stats.compare_proportions(ct.to_numpy())
            if ct.shape == (2, 2):
                odds = stats.odds_ratio(ct.to_numpy())

    fig, ax = plt.subplots(figsize=(7, 4))
    if target and target in df.columns:
        ct = pd.crosstab(df[col], df[target])
        ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.legend(title="Рецидив")
    else:
        df[col].value_counts().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"{col}", fontsize=9)
    ax.set_xlabel("")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"c_{_safe_name(col)}.png"
    fig.savefig(path)
    plt.close(fig)

    return {"категории": len(table),
            "пропуски": table.attrs.get("missing"),
            "связь с целевой": (f"{assoc.test}, p={assoc.pvalue:.4f}"
                                if assoc else "нет"),
            "отношение шансов": (f"ОШ={odds['odds_ratio']:.2f} "
                                 f"[{odds['ci_low']:.2f}; {odds['ci_high']:.2f}]"
                                 if odds else "нет"),
            "фигура": path.name}


def correlation_and_vif(df: pd.DataFrame, quant_cols: list) -> pd.DataFrame:
    """Матрица корреляций Спирмена и VIF (на медианно заполненных данных)."""
    present = [c for c in quant_cols if c in df.columns]
    data = df[present].apply(pd.to_numeric, errors="coerce")

    corr = data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Корреляции Спирмена (количественные)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "correlation_spearman.png")
    plt.close(fig)

    # VIF считаем на медианно заполненных данных как грубую оценку.
    # Добавляем константу, иначе у признаков с большим средним VIF завышен.
    filled = data.fillna(data.median())
    filled = filled.loc[:, filled.std() > 0]
    matrix = sm.add_constant(filled)
    vif_rows = []
    for i, name in enumerate(matrix.columns):
        if name == "const":
            continue
        try:
            vif = variance_inflation_factor(matrix.to_numpy(), i)
        except Exception:
            vif = np.nan
        vif_rows.append({"признак": name, "VIF": round(float(vif), 2)})
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    return vif_df


def build_report() -> str:
    """Собирает полный поколоночный отчет EDA."""
    setup_style()
    df = io.load_processed()
    roles = columns.classify(df)
    target = roles["target"]

    lines = ["# Поколоночный EDA", "",
             f"Наблюдений: {len(df)}. Целевая: {target}.", ""]

    lines.append("## Количественные показатели")
    lines.append("")
    for col in roles["quantitative"]:
        info = eda_quantitative(df, col, target)
        lines.append(f"### {col}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Категориальные показатели")
    lines.append("")
    for col in roles["categorical"]:
        info = eda_categorical(df, col, target)
        lines.append(f"### {col}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    vif_df = correlation_and_vif(df, roles["quantitative"])
    lines.append("## Мультиколлинеарность (VIF, грубая оценка)")
    lines.append("")
    lines.append(vif_df.to_markdown(index=False))
    lines.append("")

    config.ensure_dirs()
    path = config.REPORTS_DIR / "eda_columns.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    out = build_report()
    print(f"Otchet EDA: {out}")
