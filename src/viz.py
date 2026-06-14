"""Графики для статьи: единый стиль, распределение целевой, форест-график
отношений шансов, панель сравнения значимых количественных по группам рецидива.

Рисунки сохраняются в reports/figures в png (300 dpi) и svg. Подписи на русском,
без маркеров машинного слога.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import columns, config, io, stats

# Цвета групп: без рецидива и рецидив.
PALETTE = {"0.0": "#4C72B0", "1.0": "#DD8452"}
GROUP_LABELS = {0.0: "Без рецидива", 1.0: "Рецидив"}


def apply_style() -> None:
    """Единый стиль публикационного качества."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.dpi"] = config.FIGURE_DPI
    plt.rcParams["figure.dpi"] = 110


def _save(fig, name: str) -> None:
    """Сохраняет фигуру в png и svg."""
    config.ensure_dirs()
    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_target_distribution(df: pd.DataFrame, target: str):
    """Распределение целевой переменной."""
    counts = df[target].value_counts().sort_index()
    labels = [GROUP_LABELS.get(v, str(v)) for v in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values,
                  color=[PALETTE.get(str(v), "#777") for v in counts.index])
    total = counts.sum()
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 2,
                f"{n}\n({n / total * 100:.1f}%)", ha="center", fontsize=11)
    ax.set_ylabel("Число пациентов")
    ax.set_title("Распределение рецидивов ДКА")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "target_distribution")
    return fig


def _binary_contrasts(df: pd.DataFrame, target: str) -> dict:
    """Готовит бинарные контрасты для форест-графика отношений шансов."""
    g = df[target]
    contrasts = {}

    def add(label, feature_positive):
        # feature_positive - булева серия (признак есть / нет).
        sub = pd.DataFrame({"f": feature_positive, "g": g}).dropna()
        ct = pd.crosstab(sub["f"], sub["g"])
        if ct.shape == (2, 2):
            contrasts[label] = ct.to_numpy()

    if "Пол (0 - Ж, 1 - М)" in df:
        add("Мужской пол", df["Пол (0 - Ж, 1 - М)"] == 1)
    if "Невролог" in df:
        add("Осмотр невролога", df["Невролог"] == 1)
    if "Алкоголь за сутки до ДКА (0 - нет, 1 - да)" in df:
        add("Алкоголь за сутки", df["Алкоголь за сутки до ДКА (0 - нет, 1 - да)"] == 1)
    if "ХБП, С" in df:
        ckd = df["ХБП, С"].astype("object")
        add("ХБП (есть)", ckd.notna() & (ckd.astype(str) != "0"))
    retino = ("Ретинопатия (0 - нет, 1 - непролиферативная, "
              "2 - препролиферативная, 3 - пролиферативная)")
    if retino in df:
        add("Ретинопатия (есть)", pd.to_numeric(df[retino], errors="coerce") > 0)
    return contrasts


def fig_forest_odds_ratios(df: pd.DataFrame, target: str):
    """Форест-график отношений шансов для бинарных контрастов."""
    contrasts = _binary_contrasts(df, target)
    rows = []
    for label, table in contrasts.items():
        orr = stats.odds_ratio(table)
        rows.append((label, orr["odds_ratio"], orr["ci_low"], orr["ci_high"]))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(rows) + 2))
    for i, (label, orv, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color="#444", lw=1.5)
        ax.plot(orv, i, "o", color="#C44E52", ms=9)
    ax.axvline(1.0, color="#999", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("Отношение шансов рецидива (95% ДИ), лог-шкала")
    ax.set_title("Факторы, связанные с рецидивом ДКА")
    _save(fig, "forest_odds_ratios")
    return fig


def fig_significant_quantitative(df: pd.DataFrame, target: str, cols: list):
    """Панель boxplot для значимых количественных по группам рецидива."""
    n = len(cols)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        order = sorted(df[target].dropna().unique())
        sns.boxplot(data=df, x=target, y=pd.to_numeric(df[col], errors="coerce"),
                    ax=ax, hue=df[target], palette="Set2", legend=False,
                    order=order)
        sns.stripplot(data=df, x=target, y=pd.to_numeric(df[col], errors="coerce"),
                      ax=ax, color=".3", size=3, alpha=0.4, order=order)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([GROUP_LABELS.get(v, str(v)) for v in order])
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, "significant_quantitative")
    return fig


def build_article_figures() -> None:
    """Строит набор рисунков для статьи."""
    apply_style()
    df = io.load_processed()
    roles = columns.classify(df)
    target = roles["target"]

    fig_target_distribution(df, target)
    fig_forest_odds_ratios(df, target)
    significant = ["Длительность СД (лет)", "Суточная доза инсулина", "HbA1c", "ЛПВП"]
    significant = [c for c in significant if c in df.columns]
    fig_significant_quantitative(df, target, significant)


if __name__ == "__main__":
    build_article_figures()
    print("Risunki dlya statyi sohraneny v reports/figures.")
