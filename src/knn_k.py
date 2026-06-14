"""Чувствительность качества к числу соседей KNN-импутации (этап 5, доп. проверка).

В основном пайплайне KNN-импутация с n_neighbors=5 зафиксирована по ноутбуку выбора
стратегий, число соседей отдельно не перебиралось. Здесь проверяем, оптимально ли 5:
для финалистов на наборе без мультиколлинеарности считаем out-of-fold ROC-AUC при
k = 3, 5, 7, 9, 11 (импутер обучается внутри фолдов, без утечки). Смотрим лидера
(случайный лес) и среднее по четырем семействам.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io
from . import optuna_tuning as ot
from .config import RANDOM_SEED

MODELS = ["logreg", "rf", "xgb", "catboost"]
FSET = "no_collinear"
KS = [3, 5, 7, 9, 11]


def _params():
    import json
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def _set_k(pipe, k):
    keys = pipe.get_params()
    for key in ("pre__num__imp__n_neighbors", "prep__num_imputer__n_neighbors"):
        if key in keys:
            pipe.set_params(**{key: k})
            return True
    return False


def _oof_auc(df, X_train, y_train, model, feats, params, k):
    pipe = ot._build(model, feats, df, y_train, params[f"{model}|{FSET}"])
    if not _set_k(pipe, k):
        raise RuntimeError(f"не нашел KNN-импутер в пайплайне {model}")
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                  method="predict_proba")[:, 1]
    return roc_auc_score(y_train, proba)


def run() -> pd.DataFrame:
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[FSET]
    params = _params()

    rows = []
    for k in KS:
        rec = {"k": k}
        aucs = []
        for model in MODELS:
            auc = _oof_auc(df, X_train, y_train, model, feats, params, k)
            rec[model] = round(auc, 4)
            aucs.append(auc)
        rec["среднее"] = round(float(np.mean(aucs)), 4)
        rows.append(rec)
        print(f"k={k:2}  лес={rec['rf']:.4f}  среднее={rec['среднее']:.4f}")
    table = pd.DataFrame(rows)

    best_rf = table.loc[table["rf"].idxmax(), "k"]
    best_mean = table.loc[table["среднее"].idxmax(), "k"]
    print(f"\nЛучшее k по лесу: {best_rf}, по среднему: {best_mean}")

    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "knn_k.csv", index=False, encoding="utf-8-sig")
    return table


if __name__ == "__main__":
    run()
