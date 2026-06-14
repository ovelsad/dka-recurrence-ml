"""Снижение размерности и поиск фенотипов (этап 4).

Разведочный, не предсказательный анализ: PCA, t-SNE, UMAP и кластеризация
строятся на всей выборке без использования целевой переменной для обучения
(целевая нужна только для раскраски точек). Преобразования модельного пайплайна
(этапы 5-6) будут обучаться только на train.

Матрица: количественные показатели с долей пропусков не выше порога, скошенные
логарифмируем, пропуски заполняем медианой (грубо, только для визуализации; точная
импутация - этап 5), затем стандартизуем. Рисунки идут в reports/figures, краткий
отчет в reports/dimreduce.md.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from . import columns, config, io
from .config import RANDOM_SEED
from .viz import apply_style

GROUP_LABELS = {0.0: "Без рецидива", 1.0: "Рецидив"}
GROUP_COLORS = {0.0: "#4C72B0", 1.0: "#DD8452"}

# Скошенные вправо положительные показатели - логарифмируем при наличии.
LOG_COLUMNS = ["ТГ", "Глюкоза при поступлении", "Креатинин при поступлении",
               "Мочевина при поступлении", "Суточная доза инсулина"]


def build_matrix(df: pd.DataFrame, max_missing: float = 0.4):
    """Готовит стандартизованную числовую матрицу для снижения размерности."""
    quant = columns.classify(df)["quantitative"]
    data = df[quant].apply(pd.to_numeric, errors="coerce")

    # Отбор по доле пропусков.
    keep = [c for c in quant if data[c].isna().mean() <= max_missing]
    data = data[keep].copy()

    # Логарифм для скошенных (только положительные значения).
    for c in LOG_COLUMNS:
        if c in data.columns and (data[c].dropna() > 0).all():
            data[c] = np.log1p(data[c])

    # Медианная импутация (грубо, только для визуализации) и стандартизация.
    data = data.fillna(data.median())
    scaled = StandardScaler().fit_transform(data.to_numpy())
    return scaled, keep


def correlation_heatmaps(df: pd.DataFrame, min_periods: int = 20) -> dict:
    """Матрицы корреляций количественных показателей на сырых данных.

    Корреляции считаем на данных БЕЗ заполнения пропусков, попарно (pairwise):
    каждая пара коэффициентов оценивается по наблюдениям, где оба показателя
    присутствуют. min_periods отсекает пары со слишком малым перекрытием. Строим
    две матрицы: Спирмена (ранговая, устойчива к скошенности и выбросам) и Пирсона
    (линейная). Рисунки идут в reports/figures.
    """
    quant = columns.classify(df)["quantitative"]
    data = df[quant].apply(pd.to_numeric, errors="coerce")
    short = {c: (c if len(c) <= 22 else c[:20] + "..") for c in quant}
    data = data.rename(columns=short)

    config.ensure_dirs()
    out = {}
    for method, fname, title in [
        ("spearman", "corr_spearman.png",
         "Корреляции Спирмена (количественные, сырые данные)"),
        ("pearson", "corr_pearson.png",
         "Корреляции Пирсона (количественные, сырые данные)"),
    ]:
        corr = data.corr(method=method, min_periods=min_periods)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                    annot=True, fmt=".2f", annot_kws={"size": 6}, linewidths=0.4,
                    linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / fname, bbox_inches="tight")
        plt.close(fig)
        out[method] = corr
    return out


def _target_colors(target_series):
    """Цвета точек по группам целевой, пропуски серым."""
    return [GROUP_COLORS.get(v, "#cccccc") for v in target_series]


def _legend_handles():
    import matplotlib.patches as mpatches
    return [mpatches.Patch(color=c, label=GROUP_LABELS[v])
            for v, c in GROUP_COLORS.items()]


def run_pca(X, target_series) -> dict:
    """PCA: график каменистой осыпи, проекция PC1-PC2, вклады признаков."""
    pca = PCA(random_state=RANDOM_SEED)
    scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    # График доли объясненной дисперсии.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, "o-", color="#4C72B0")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Накопленная дисперсия, %")
    ax.set_title("PCA: накопленная объясненная дисперсия")
    ax.axhline(80, color="#999", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pca_scree.png", bbox_inches="tight")
    plt.close(fig)

    # Проекция на первые две компоненты.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores[:, 0], scores[:, 1], c=_target_colors(target_series),
               s=28, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title("PCA: проекция пациентов")
    ax.legend(handles=_legend_handles())
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pca_scatter.png", bbox_inches="tight")
    plt.close(fig)

    return {"explained": evr, "scores": scores, "pca": pca}


def run_tsne(X, target_series):
    """t-SNE: двумерная проекция, только для визуализации структуры."""
    emb = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED,
               init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=_target_colors(target_series),
               s=28, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_title("t-SNE (визуализация, расстояния не интерпретируем)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(handles=_legend_handles())
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "tsne_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return emb


def run_umap(X, target_series):
    """UMAP: двумерная проекция, если библиотека доступна."""
    if not HAS_UMAP:
        return None
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    emb = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=_target_colors(target_series),
               s=28, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_title("UMAP (визуализация структуры)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(handles=_legend_handles())
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "umap_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return emb


def run_clustering(X, scores, target_series) -> dict:
    """KMeans с подбором числа кластеров по силуэту, сверка с целевой."""
    sil = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
        sil[k] = float(silhouette_score(X, labels))
    best_k = max(sil, key=sil.get)
    labels = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10).fit_predict(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=labels, palette="tab10",
                    ax=ax, s=28, alpha=0.8, legend="full")
    ax.set_title(f"KMeans на PCA, k={best_k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "clusters_pca.png", bbox_inches="tight")
    plt.close(fig)

    crosstab = pd.crosstab(pd.Series(labels, name="кластер"),
                           pd.Series(target_series.values, name="рецидив"))
    return {"silhouette": sil, "best_k": best_k, "crosstab": crosstab}


def build_all() -> str:
    apply_style()
    df = io.load_processed()
    target = columns.classify(df)["target"]
    target_series = df[target]

    X, used = build_matrix(df)
    pca_res = run_pca(X, target_series)
    run_tsne(X, target_series)
    umap_emb = run_umap(X, target_series)
    clust = run_clustering(X, pca_res["scores"], target_series)

    # Вклады признаков в первые две компоненты.
    comps = pd.DataFrame(pca_res["pca"].components_[:2].T, index=used,
                         columns=["PC1", "PC2"])

    lines = ["# Снижение размерности (этап 4)", "",
             f"Использовано признаков: {len(used)} (доля пропусков не выше 40%).",
             f"Список: {used}", "",
             "## PCA",
             f"Доля дисперсии по компонентам, %: "
             f"{[round(v * 100, 1) for v in pca_res['explained'][:8]]}",
             f"Накоплено первыми 2: {sum(pca_res['explained'][:2]) * 100:.1f}%, "
             f"первыми 5: {sum(pca_res['explained'][:5]) * 100:.1f}%", "",
             "Топ вкладов в PC1:",
             comps["PC1"].abs().sort_values(ascending=False).head(6).to_string(),
             "", "Топ вкладов в PC2:",
             comps["PC2"].abs().sort_values(ascending=False).head(6).to_string(),
             "", "## Кластеризация",
             f"Силуэт по k: {clust['silhouette']}",
             f"Лучшее k: {clust['best_k']}", "",
             "Кластеры против рецидива:",
             clust["crosstab"].to_string(),
             "", "UMAP построен." if umap_emb is not None else "UMAP недоступен.",
             "", "Замечание: t-SNE и UMAP - визуализация структуры, абсолютные "
             "расстояния и размеры кластеров не интерпретируем. Импутация медианой "
             "грубая, после этапа 5 анализ повторим на честно заполненных данных."]

    config.ensure_dirs()
    path = config.REPORTS_DIR / "dimreduce.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    out = build_all()
    print(f"Otchet dimreduce: {out}")
