"""Влияние преобразований количественных признаков (мини-этап).

Сравниваем три варианта предобработки для масштабочувствительных моделей
(логистическая регрессия, метод опорных векторов, k ближайших соседей):
- none: только стандартизация;
- manual: логарифм и Йео-Джонсон по схеме reports/transformations.md, затем
  стандартизация;
- yeojohnson_all: степенное преобразование Йео-Джонсона ко всем количественным,
  затем стандартизация.

Деревья инвариантны к монотонным преобразованиям, поэтому здесь их не берем.
Оценка - ROC-AUC по out-of-fold с бутстрэп-ДИ, без утечки (преобразования и
импутация обучаются на train внутри фолдов). Тест не трогаем: это побочное
сравнение, его место на кросс-валидации.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   PowerTransformer, StandardScaler)
from sklearn.svm import SVC

from . import config, features, io
from .config import RANDOM_SEED
from .evaluation import bootstrap_metrics

# Схема из reports/transformations.md.
LOG_COLS = ["ТГ", "Общий холестерин", "Лактат при поступлении", "HbA1c", "ЛПНП",
            "ЛПВП", "Глюкоза при поступлении", "Мочевина при поступлении",
            "Креатинин при поступлении"]
YJ_COLS = ["ВЕ при поступлении", "Длительность СД (лет)"]

# Масштабочувствительные модели (на них преобразования и влияют) и деревья
# (инвариантны к монотонным преобразованиям, включены для полноты таблицы).
MODELS = ["logreg", "svm", "knn", "rf", "lgbm", "catboost", "xgb"]
FSETS = ["significant", "no_collinear"]
VARIANTS = ["none", "manual", "yeojohnson_all"]


def _estimator(model):
    if model == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RANDOM_SEED)
    if model == "svm":
        return SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=7)
    if model == "rf":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=RANDOM_SEED, n_jobs=-1)
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED,
                              verbose=-1)
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, verbose=0, random_state=RANDOM_SEED,
                                  auto_class_weights="Balanced")
    if model == "xgb":
        from xgboost import XGBClassifier
        # scale_pos_weight - аналог class_weight="balanced" у остальных моделей:
        # отношение числа наблюдений отрицательного класса к положительному.
        return XGBClassifier(n_estimators=400, random_state=RANDOM_SEED,
                             eval_metric="logloss", scale_pos_weight=149 / 69)
    raise ValueError(model)


def _numeric_pipe(variant, quant):
    """Числовой блок: импутация KNN, преобразование по варианту, стандартизация."""
    steps = [("imp", KNNImputer(n_neighbors=5))]
    if variant == "manual":
        log_idx = [i for i, c in enumerate(quant) if c in LOG_COLS]
        yj_idx = [i for i, c in enumerate(quant) if c in YJ_COLS]
        none_idx = [i for i, c in enumerate(quant)
                    if c not in LOG_COLS and c not in YJ_COLS]
        trans = ColumnTransformer([
            ("log", FunctionTransformer(np.log1p), log_idx),
            ("yj", PowerTransformer(method="yeo-johnson"), yj_idx),
            ("none", "passthrough", none_idx),
        ])
        steps.append(("trans", trans))
    elif variant == "yeojohnson_all":
        steps.append(("yj", PowerTransformer(method="yeo-johnson")))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _build(df, feats, model, variant):
    quant, cat = features.column_types(df, feats)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", _numeric_pipe(variant, quant), quant),
                             ("cat", cat_pipe, cat)])
    return Pipeline([("pre", pre), ("clf", _estimator(model))])


def run() -> pd.DataFrame:
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    sets = features.feature_sets(df)

    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    for model in MODELS:
        for fset in FSETS:
            feats = sets[fset]
            for variant in VARIANTS:
                pipe = _build(df, feats, model, variant)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                              method="predict_proba")[:, 1]
                m = bootstrap_metrics(y_train.to_numpy(), proba)
                roc, lo, hi = m["ROC-AUC"]
                rows.append({"модель": model, "набор": fset, "преобразование": variant,
                             "ROC_AUC": roc, "ДИ": f"[{lo}; {hi}]"})
                print(f"{model:7} {fset:13} {variant:15} ROC-AUC={roc} [{lo}; {hi}]")
    table = pd.DataFrame(rows)
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "transforms.csv", index=False,
                 encoding="utf-8-sig")
    return table


if __name__ == "__main__":
    run()
