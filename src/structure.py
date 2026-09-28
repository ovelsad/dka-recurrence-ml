"""Шаг 4: проверка импутации и структура данных без учителя

Все считаем только на train. Импутер и масштаб обучаются на train, исход в
импутации не участвует.

- Сохранение распределений: заполненные значения признака сравниваем с
  наблюдаемыми критерием Колмогорова-Смирнова. Чем меньше статистика KS, тем
  больше заполненные похожи на настоящие. Медиана ставит всем одно значение и
  дает пик, это видно по KS
- Пропуск против исхода: связан ли сам факт пропуска с рецидивом
- Облака PCA: пациенты с заполненными значениями не должны уходить в отдельное
  облако на плоскости двух главных компонент
- Структура: доля дисперсии по главным компонентам, t-SNE и k-means с выбором
  числа кластеров по силуэту, затем доля рецидива по кластерам
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from . import config
from .eda import GROUP_COLORS, GROUP_NAMES, INK, short_name
from .models import NUMERIC, numeric_block
from .stats import compare_categorical, p_text

MIN_FILLED = 5  # KS считаем, если заполнено хотя бы столько значений признака
CLUSTER_RANGE = range(2, 7)  # кандидаты числа кластеров k-means
TSNE_PERPLEXITY = 30  # стандартное значение, при 229 пациентах в допустимых пределах
# Цвета групп по числу заполненных значений у пациента: от без пропусков к многим
FILL_COLORS = ["#2a78d6", "#eda100", "#e34948"]
FILL_GROUPS = ["без пропусков", "заполнено 1-3", "заполнено 4 и больше"]
# Цвета кластеров отличаются от цветов исхода, чтобы кластер не читался как исход
CLUSTER_COLORS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300", "#e34948"]


def imputed(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """Числовые признаки train после масштаба и импутации методом method"""
    block = numeric_block(method).fit(X[NUMERIC])
    return pd.DataFrame(block.transform(X[NUMERIC]), index=X.index, columns=NUMERIC)


def ks_table(X: pd.DataFrame, methods: list) -> pd.DataFrame:
    """Статистика KS между заполненными и наблюдаемыми значениями по признакам

    Берем количественные признаки, у которых заполнено не меньше MIN_FILLED
    значений. Последняя строка - медиана по признакам
    """
    missing = X[config.QUANTITATIVE].isna()
    columns = [c for c in config.QUANTITATIVE if missing[c].sum() >= MIN_FILLED]
    table = pd.DataFrame(index=[short_name(c) for c in columns])
    for method in methods:
        filled = imputed(X, method)
        table[method] = [
            scipy_stats.ks_2samp(
                filled.loc[~missing[c], c], filled.loc[missing[c], c]
            ).statistic
            for c in columns
        ]
    table.loc["медиана по признакам"] = table[methods].median()
    table["заполнено"] = [int(missing[c].sum()) for c in columns] + [np.nan]
    return table


def missing_outcome(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Связь пропуска в признаке с исходом на train, критерий по протоколу

    Для каждого признака с пропусками сравниваем долю рецидива у пациентов с
    пропуском и без него (stats.compare_categorical: хи-квадрат Пирсона или
    точный критерий Фишера, ОШ пропуска). Если пропуск связан с исходом, то
    заполненное значение несет сведения об исходе
    """
    rows = []
    for column in X.columns[X.isna().any()]:
        missing = X[column].isna()
        test = compare_categorical(missing.astype(int), y)
        rows.append(
            {
                "признак": short_name(column),
                "пропусков": int(missing.sum()),
                "рецидив среди пропусков": y[missing].mean(),
                "рецидив среди заполненных": y[~missing].mean(),
                "ОШ": test["or"],
                "ОШ ДИ низ": test["or_lo"],
                "ОШ ДИ верх": test["or_hi"],
                "критерий": test["test"],
                "p": test["p"],
            }
        )
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def plot_missing_outcome(table: pd.DataFrame, overall: float) -> plt.Figure:
    """Доля рецидива среди пропусков и среди заполненных по признакам

    overall - доля рецидива на train, вертикальный пунктир
    """
    positions = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(table) + 1.6))
    for column, color, marker in [
        ("рецидив среди заполненных", GROUP_COLORS[0], "o"),
        ("рецидив среди пропусков", GROUP_COLORS[1], "s"),
    ]:
        ax.scatter(
            table[column], positions, color=color, marker=marker, s=40, label=column
        )
    ax.hlines(
        positions,
        table[["рецидив среди заполненных", "рецидив среди пропусков"]].min(axis=1),
        table[["рецидив среди заполненных", "рецидив среди пропусков"]].max(axis=1),
        color="lightgray",
        lw=2,
        zorder=0,
    )
    ax.axvline(overall, color="gray", lw=1, ls="--", label="доля рецидива на train")
    labels = [
        f"{name} (p {p_text(p)})" for name, p in zip(table["признак"], table["p"])
    ]
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set(xlabel="доля рецидива", title="Доля рецидива среди пропусков и заполненных")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fill_group(X: pd.DataFrame) -> pd.Series:
    """Группа пациента по числу заполненных числовых значений"""
    count = X[NUMERIC].isna().sum(axis=1)
    return pd.cut(count, bins=[-1, 0, 3, np.inf], labels=FILL_GROUPS)


def plot_pca_clouds(X: pd.DataFrame, methods: list) -> plt.Figure:
    """Пациенты на плоскости двух главных компонент для каждого метода импутации"""
    groups = fill_group(X)
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.6))
    for ax, method in zip(np.atleast_1d(axes), methods):
        points = PCA(n_components=2, random_state=config.RANDOM_SEED).fit_transform(
            imputed(X, method)
        )
        for color, group in zip(FILL_COLORS, FILL_GROUPS):
            mask = (groups == group).to_numpy()
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=18,
                color=color,
                alpha=0.8,
                label=f"{group} (n={mask.sum()})",
            )
        ax.set(title=method, xlabel="ГК 1", ylabel="ГК 2")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Облака PCA по числу заполненных значений у пациента", color=INK)
    fig.tight_layout()
    return fig


def pca_variance(Z: pd.DataFrame) -> pd.DataFrame:
    """Доля дисперсии по главным компонентам и накопленная доля"""
    pca = PCA(random_state=config.RANDOM_SEED).fit(Z)
    share = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "компонента": np.arange(1, share.size + 1),
            "доля дисперсии": share,
            "накопленная": np.cumsum(share),
        }
    )


def plot_pca_variance(variance: pd.DataFrame) -> plt.Figure:
    """Доля дисперсии по компонентам (столбцы) и накопленная доля (линия)"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        variance["компонента"],
        variance["доля дисперсии"],
        color=GROUP_COLORS[0],
        width=0.6,
        label="компонента",
    )
    ax.plot(
        variance["компонента"],
        variance["накопленная"],
        color=INK,
        marker="o",
        ms=4,
        lw=1.5,
        label="накопленная",
    )
    ax.axhline(0.8, color="gray", lw=1, ls="--")
    ax.set(
        xlabel="главная компонента",
        ylabel="доля дисперсии",
        title="Доля дисперсии по главным компонентам",
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_by_outcome(
    points: np.ndarray, y: pd.Series, title: str, axis: str
) -> plt.Figure:
    """Точки двумерной проекции, окрашенные по исходу, axis - имя осей (ГК, t-SNE)"""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for group in (0, 1):
        mask = (y == group).to_numpy()
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=18,
            alpha=0.8,
            color=GROUP_COLORS[group],
            label=f"{GROUP_NAMES[group]} (n={mask.sum()})",
        )
    ax.set(title=title, xlabel=f"{axis} 1", ylabel=f"{axis} 2")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def tsne_points(Z: pd.DataFrame) -> np.ndarray:
    """Двумерная проекция t-SNE

    Расстояния между группами точек на t-SNE не несут смысла, смотрим только на
    то, собираются ли пациенты одного исхода вместе
    """
    return TSNE(
        perplexity=TSNE_PERPLEXITY, random_state=config.RANDOM_SEED
    ).fit_transform(Z)


def kmeans_labels(Z: pd.DataFrame, k: int) -> np.ndarray:
    """Номера кластеров k-means (с 1) для каждого пациента"""
    model = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED)
    return model.fit_predict(Z) + 1


def silhouette_table(Z: pd.DataFrame) -> pd.DataFrame:
    """Силуэт k-means для каждого числа кластеров из CLUSTER_RANGE

    Силуэт от -1 до 1: чем выше, тем четче кластеры отделены друг от друга
    """
    rows = [
        {"кластеров": k, "силуэт": silhouette_score(Z, kmeans_labels(Z, k))}
        for k in CLUSTER_RANGE
    ]
    return pd.DataFrame(rows)


def plot_silhouette(silhouettes: pd.DataFrame) -> plt.Figure:
    """Силуэт k-means по числу кластеров"""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(
        silhouettes["кластеров"],
        silhouettes["силуэт"],
        color=GROUP_COLORS[0],
        marker="o",
        ms=8,
        lw=2,
    )
    ax.set(
        title="Силуэт k-means",
        xlabel="число кластеров",
        ylabel="средний силуэт",
        xticks=list(silhouettes["кластеров"]),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def cluster_outcome(labels: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Доля рецидива по кластерам и связь кластера с исходом

    Критерий выбирается по протоколу (stats.compare_categorical), его название и
    p лежат в attrs["критерий"] и attrs["p"]
    """
    clusters = pd.Series(labels, index=y.index, name="кластер")
    table = pd.crosstab(clusters, y)
    summary = pd.DataFrame(
        {
            "пациентов": table.sum(axis=1),
            "рецидивов": table[1],
            "доля рецидива": table[1] / table.sum(axis=1),
        }
    )
    test = compare_categorical(clusters, y)
    summary.attrs["критерий"], summary.attrs["p"] = test["test"], test["p"]
    return summary


def plot_clusters(
    points: np.ndarray, labels: np.ndarray, summary: pd.DataFrame
) -> plt.Figure:
    """Кластеры k-means на плоскости двух главных компонент

    В легенде размер кластера и доля рецидива в нем
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for color, cluster in zip(CLUSTER_COLORS, summary.index):
        mask = labels == cluster
        size, share = (
            summary.at[cluster, "пациентов"],
            summary.at[cluster, "доля рецидива"],
        )
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=18,
            alpha=0.8,
            color=color,
            label=f"кластер {cluster}: n={size}, рецидив {share:.0%}",
        )
    ax.set(title="Кластеры k-means на плоскости PCA", xlabel="ГК 1", ylabel="ГК 2")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
