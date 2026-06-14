"""Стратегии заполнения пропусков и сравнение распределений (этап 5).

Сравниваем четыре варианта:
- none: пропуски не трогаем (для CatBoost, он обрабатывает NaN сам);
- median_mode: медиана для количественных, мода для категориальных;
- knn: заполнение по похожим пациентам (на стандартизованных данных);
- mice: множественная импутация моделями (IterativeImputer).

Категориальные во всех вариантах (кроме none) заполняем модой - KNN и MICE на
категориях работают плохо. Целевую переменную не трогаем никогда. Готовые наборы
сохраняем в data/processed для последующего обучения и сравнения качества.

ВАЖНО про утечку: здесь импутация на всей выборке нужна только чтобы оценить сдвиг
распределений (описательно). В модельном пайплайне (этапы 6-7) импутер обучается
только на train внутри фолдов кросс-валидации.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import columns, config, io
from .config import RANDOM_SEED

STRATEGIES = ["none", "median_mode", "knn", "mice"]

# Показатели с высокой долей пропусков - на них смотрим сдвиг распределений.
WATCH_COLS = ["HbA1c", "Натрий при поступлении", "Калий при поступлении",
              "Лактат при поступлении"]


def _impute_quantitative(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Импутация количественной части по выбранной стратегии."""
    arr = data.to_numpy(dtype=float)
    if strategy == "median_mode":
        filled = SimpleImputer(strategy="median").fit_transform(arr)
    elif strategy == "knn":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr)
        imputed = KNNImputer(n_neighbors=5).fit_transform(scaled)
        filled = scaler.inverse_transform(imputed)
    elif strategy == "mice":
        filled = IterativeImputer(random_state=RANDOM_SEED, max_iter=15).fit_transform(arr)
    else:
        raise ValueError(strategy)
    return pd.DataFrame(filled, columns=data.columns, index=data.index)


def impute_dataset(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Возвращает датасет с заполненными пропусками по стратегии."""
    if strategy == "none":
        return df.copy()

    roles = columns.classify(df)
    quant = [c for c in roles["quantitative"] if c in df.columns]
    cat = [c for c in roles["categorical"] if c in df.columns]

    result = df.copy()
    result[quant] = _impute_quantitative(df[quant].apply(pd.to_numeric, errors="coerce"),
                                          strategy)

    if cat:
        cat_obj = df[cat].astype("object")
        filled = SimpleImputer(strategy="most_frequent").fit_transform(cat_obj)
        result[cat] = pd.DataFrame(filled, columns=cat, index=df.index)
    return result


def build_all_variants() -> dict:
    """Строит и сохраняет все варианты импутации в data/processed."""
    df = io.load_processed()
    saved = {}
    for strategy in STRATEGIES:
        imputed = impute_dataset(df, strategy)
        io.save_processed(imputed, f"dka_imputed_{strategy}")
        saved[strategy] = imputed
    return saved


def compare_distributions(variants: dict) -> pd.DataFrame:
    """Сравнивает распределения количественных до и после импутации.

    Для каждой стратегии и показателя: сдвиг медианы и SD, KS-критерий между
    исходными наблюдаемыми значениями и полным столбцом после импутации.
    """
    base = variants["none"]
    roles = columns.classify(base)
    quant = [c for c in roles["quantitative"] if c in base.columns]

    rows = []
    for col in quant:
        observed = pd.to_numeric(base[col], errors="coerce").dropna()
        for strategy in STRATEGIES:
            if strategy == "none":
                continue
            full = pd.to_numeric(variants[strategy][col], errors="coerce").dropna()
            ks = sp_stats.ks_2samp(observed, full)
            rows.append({
                "показатель": col,
                "стратегия": strategy,
                "медиана_до": round(observed.median(), 2),
                "медиана_после": round(full.median(), 2),
                "SD_до": round(observed.std(ddof=1), 2),
                "SD_после": round(full.std(ddof=1), 2),
                "KS_p": round(float(ks.pvalue), 4),
            })
    return pd.DataFrame(rows)


def plot_watch_distributions(variants: dict) -> None:
    """Накладывает распределения до и после импутации для высокопропусковых."""
    base = variants["none"]
    for col in WATCH_COLS:
        if col not in base.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        observed = pd.to_numeric(base[col], errors="coerce").dropna()
        sns.kdeplot(observed, ax=ax, label="исходные (наблюдаемые)", lw=2.5,
                    color="black")
        for strategy in ["median_mode", "knn", "mice"]:
            full = pd.to_numeric(variants[strategy][col], errors="coerce").dropna()
            sns.kdeplot(full, ax=ax, label=strategy)
        ax.set_title(f"Распределение до и после импутации: {col}", fontsize=11)
        ax.set_xlabel("")
        ax.legend()
        fig.tight_layout()
        safe = col.replace(" ", "_").replace("/", "_")[:40]
        fig.savefig(config.FIGURES_DIR / f"impute_{safe}.png", bbox_inches="tight")
        plt.close(fig)


def build_report() -> str:
    config.ensure_dirs()
    variants = build_all_variants()
    comparison = compare_distributions(variants)
    plot_watch_distributions(variants)

    # Сводка: средний сдвиг распределения по стратегиям (медиана KS_p, чем выше,
    # тем меньше искажение).
    summary = comparison.groupby("стратегия")["KS_p"].median().round(3)

    lines = ["# Сравнение стратегий импутации (этап 5)", "",
             "Сохранены наборы: " + ", ".join(f"dka_imputed_{s}" for s in STRATEGIES),
             "", "## Медианный KS_p по стратегиям",
             "Выше KS_p - меньше искажение распределения относительно наблюдаемых "
             "значений.", "", summary.to_string(), "",
             "## Сдвиг распределений по показателям", "",
             comparison.to_markdown(index=False), "",
             "Замечание: импутация здесь на всей выборке, только для оценки сдвига. "
             "В обучении импутер обучаем на train внутри фолдов."]
    path = config.REPORTS_DIR / "imputation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    out = build_report()
    print(f"Otchet imputation: {out}")
