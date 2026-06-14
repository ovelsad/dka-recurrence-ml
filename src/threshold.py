"""Оптимизация порога классификации (отдельный этап).

Порог 0.5 редко оптимален для редкого важного класса (рецидивы). Подбираем порог
на обучающей выборке по out-of-fold предсказаниям (без утечки), затем применяем
зафиксированный порог к отложенному тесту.

Критерии выбора порога:
- Youden (J = чувствительность + специфичность - 1): баланс;
- F2: F-мера с весом полноты выше точности (пропуск рецидива дороже ложной тревоги);
- порог под целевую чувствительность (например не ниже 0.85) с максимальной
  специфичностью - клинический подход "не пропустить группу риска".

Считаем чувствительность, специфичность, PPV (доля истинных среди тревог) и NPV
(доля здоровых среди отрицательных) - они важны врачу.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score

from . import config


def metrics_at(y, proba, t: float) -> dict:
    """Метрики при заданном пороге."""
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {"порог": round(t, 3), "чувств.": round(sens, 3),
            "специф.": round(spec, 3), "PPV": round(ppv, 3),
            "NPV": round(npv, 3),
            "F2": round(fbeta_score(y, pred, beta=2, zero_division=0), 3)}


def select_thresholds(y, proba, sens_target: float = 0.85) -> dict:
    """Подбор порогов по разным критериям на основе переданных вероятностей."""
    grid = np.linspace(0.02, 0.98, 193)
    table = pd.DataFrame([metrics_at(y, proba, t) for t in grid])

    youden = table.loc[(table["чувств."] + table["специф."] - 1).idxmax(), "порог"]
    f2 = table.loc[table["F2"].idxmax(), "порог"]

    # Порог под целевую чувствительность с максимальной специфичностью.
    # .loc по метке из idxmax: ok - подмножество, позиционный .iloc тут давал бы
    # неверную строку или выход за границы.
    ok = table[table["чувств."] >= sens_target]
    sens_thr = ok.loc[ok["специф."].idxmax(), "порог"] if not ok.empty \
        else grid[0]

    return {"Youden": float(youden), "F2": float(f2),
            f"чувств.>={sens_target}": float(sens_thr)}


def plot_threshold_curves(y, proba, chosen: dict, name: str) -> None:
    """Чувствительность и специфичность по порогу, с отметками выбранных."""
    grid = np.linspace(0.02, 0.98, 193)
    sens = [metrics_at(y, proba, t)["чувств."] for t in grid]
    spec = [metrics_at(y, proba, t)["специф."] for t in grid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, sens, label="чувствительность", color="#C44E52")
    ax.plot(grid, spec, label="специфичность", color="#4C72B0")
    colors = ["#55A868", "#8172B3", "#CCB974"]
    for (label, t), c in zip(chosen.items(), colors):
        ax.axvline(t, ls="--", color=c, label=f"{label} ({t:.2f})")
    ax.axvline(0.5, ls=":", color="#999", label="порог 0.5")
    ax.set_xlabel("Порог вероятности")
    ax.set_ylabel("Значение метрики")
    ax.set_title(f"Чувствительность и специфичность по порогу ({name})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(config.FIGURES_DIR / f"threshold_{name}.png", bbox_inches="tight")
    plt.close(fig)


# Подбор порога на тюнингованных моделях и его проверку на тесте делает ноутбук 09
# (на Optuna-параметрах из tuning_optuna_params.json). Здесь только утилиты:
# select_thresholds, metrics_at, plot_threshold_curves. Их же использует сервис
# (src/service_model.py).
