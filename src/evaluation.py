"""Финальная оценка: калибровка, кривые принятия решений, тест (этап 7).

Содержит:
- bootstrap_metrics: точечные оценки и 95% ДИ метрик бутстрэпом;
- decision_curve / plot_decision_curve: чистая польза по диапазону порогов;
- calibration_compare: сравнение сырых и калиброванных вероятностей (Платт,
  изотоническая), калибровочные кривые и Brier;
- evaluate_on_test: обучение лучшей модели на train, честная оценка на отложенном
  тесте.

Отложенный тест трогаем один раз, в самом конце.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

from . import config, features, io
from .config import RANDOM_SEED


def bootstrap_metrics(y_true, proba, n_boot: int = 2000, threshold: float = 0.5) -> dict:
    """Точечные оценки и 95% ДИ метрик через бутстрэп."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)

    def compute(idx):
        yt, pr = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            return None
        pred = (pr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        return {
            "ROC-AUC": roc_auc_score(yt, pr),
            "PR-AUC": average_precision_score(yt, pr),
            "Чувствительность": tp / (tp + fn) if (tp + fn) else np.nan,
            "Специфичность": tn / (tn + fp) if (tn + fp) else np.nan,
            "Brier": brier_score_loss(yt, pr),
        }

    point = compute(np.arange(n))
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        res = compute(idx)
        if res:
            for k, v in res.items():
                boots[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = (round(float(v), 3), round(float(lo), 3), round(float(hi), 3))
    return out


def decision_curve(y_true, proba, thresholds=None):
    """Чистая польза модели, стратегий лечить всех и не лечить никого."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    prevalence = y_true.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.6, 60)

    nb_model, nb_all = [], []
    for pt in thresholds:
        w = pt / (1 - pt)
        pred = (proba >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prevalence - (1 - prevalence) * w)
    return thresholds, np.array(nb_model), np.array(nb_all)


def plot_decision_curve(y_true, proba, name: str = "model") -> None:
    """Сохраняет график кривых принятия решений."""
    thr, nb_model, nb_all = decision_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, nb_model, label="модель", color="#C44E52", lw=2)
    ax.plot(thr, nb_all, label="лечить всех", color="#4C72B0", ls="--")
    ax.axhline(0, label="не лечить никого", color="#999", ls=":")
    ax.set_xlabel("Пороговая вероятность")
    ax.set_ylabel("Чистая польза")
    ax.set_title("Кривые принятия решений")
    ax.set_ylim(min(-0.02, np.nanmin(nb_model)), max(nb_model) * 1.2 + 0.01)
    ax.legend()
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(config.FIGURES_DIR / f"decision_curve_{name}.png", bbox_inches="tight")
    plt.close(fig)


def calibration_compare(base_estimator, X_train, y_train, X_test, y_test,
                        name: str = "model") -> pd.DataFrame:
    """Сравнивает сырые и калиброванные вероятности (Платт и изотоническая)."""
    results = []
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="идеальная калибровка")

    variants = {
        "без калибровки": base_estimator,
        "Платт (сигмоида)": CalibratedClassifierCV(base_estimator, method="sigmoid", cv=5),
        "изотоническая": CalibratedClassifierCV(base_estimator, method="isotonic", cv=5),
    }
    for label, est in variants.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train, y_train)
            proba = est.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        results.append({"вариант": label, "Brier": round(brier, 3),
                        "ROC-AUC": round(auc, 3)})
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5,
                                                strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=f"{label} (Brier {brier:.3f})")

    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Доля рецидивов")
    ax.set_title("Калибровочные кривые")
    ax.legend(fontsize=9)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(config.FIGURES_DIR / f"calibration_{name}.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(results)


def evaluate_on_test_tuned(model: str, fset: str):
    """Оценка на тесте с финальными гиперпараметрами из Optuna (ноутбук 7).

    Берет параметры из reports/tables/tuning_optuna_params.json (ключ "модель|набор"),
    обучает пайплайн на train, один раз оценивает на отложенном тесте. Калибровку
    обучает на train, порог сюда не вносим (он в ноутбуке 9).
    """
    import json

    from . import optuna_tuning as ot

    df = io.load_processed()
    X_train, X_test, y_train, y_test = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    params = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8")
    )[f"{model}|{fset}"]

    pipe = ot._build(model, feats, df, y_train, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train[feats], y_train)
        proba = pipe.predict_proba(X_test[feats])[:, 1]

    metrics = bootstrap_metrics(y_test, proba)
    plot_decision_curve(y_test, proba, name=model)
    calib = calibration_compare(pipe, X_train[feats], y_train, X_test[feats], y_test,
                                name=model)

    lines = [f"# Оценка на отложенном тесте (Optuna): {model}, набор {fset}", "",
             f"Тест: {len(y_test)} пациентов. Гиперпараметры: {params}", "",
             "## Метрики на тесте (точка и 95% ДИ бутстрэп)", ""]
    for k, (pt, lo, hi) in metrics.items():
        lines.append(f"- {k}: {pt} [{lo}; {hi}]")
    lines += ["", "## Калибровка (на тесте)", "", calib.to_markdown(index=False)]
    config.ensure_dirs()
    (config.REPORTS_DIR / f"test_eval_{model}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return metrics, calib


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    fset = sys.argv[2] if len(sys.argv) > 2 else "no_collinear"
    metrics, calib = evaluate_on_test_tuned(model, fset)
    print(f"Тест для {model}, набор {fset}:")
    for k, (p, lo, hi) in metrics.items():
        print(f"  {k}: {p} [{lo}; {hi}]")
    print(calib.to_string(index=False))
