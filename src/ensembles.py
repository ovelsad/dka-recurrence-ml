"""Стекинг финалистов (этап 11).

Проверяем, дает ли стекинг прирост над лучшими одиночными моделями. Базовые модели
- восемь финалистов (логрег, лес, XGBoost, CatBoost на наборах significant и
no_collinear) на тюнингованных гиперпараметрах из ноутбука 07. Мета-модель -
логистическая регрессия над out-of-fold предсказаниями базовых (без утечки).

Перебираем все комбинации базовых моделей (от двух штук). Чтобы перебор был
выполнимым, оцениваем комбинации экономно: один раз считаем OOF-вероятности каждого
финалиста на train, затем для каждой комбинации обучаем мета-логрег на этих
OOF-столбцах по той же кросс-валидации. Скрининг использует одни и те же фолды для
OOF базовых моделей и для мета-CV, поэтому он слегка оптимистичен и нужен только для
отбора комбинации. Несмещенную оценку дает настоящий StackingClassifier (с внутренней
кросс-валидацией), который собираем для лучшей комбинации и проверяем на отложенном
тесте.

Балансировку класса в мета-модель не вносим: мета учится на вероятностях, а порог
выбирается отдельно (ноутбук 09) и на отбор комбинаций не влияет (сравниваем по
ROC-AUC, она от порога не зависит).
"""

import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io
from . import optuna_tuning as ot
from .config import RANDOM_SEED

MODELS = ["logreg", "rf", "xgb", "catboost"]
SETS = ["significant", "no_collinear"]
FINALISTS = [(m, fs) for fs in SETS for m in MODELS]


def _label(model, fset):
    return f"{model}|{fset}"


def _meta():
    return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)


def _load_params():
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def base_oof():
    """OOF-вероятности каждого финалиста на train и их одиночные ROC-AUC."""
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    y = y_train.to_numpy()
    params = _load_params()
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

    oof, single = {}, {}
    for model, fset in FINALISTS:
        feats = features.feature_sets(df)[fset]
        pipe = ot._build(model, feats, df, y_train, params[f"{model}|{fset}"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                  method="predict_proba")[:, 1]
        lab = _label(model, fset)
        oof[lab] = p
        single[lab] = roc_auc_score(y, p)
        print(f"{lab:22} OOF ROC-AUC={single[lab]:.3f}")
    return oof, single, y


def screen_combinations(oof: dict, y) -> pd.DataFrame:
    """Все комбинации базовых моделей (от двух): мета-логрег по OOF-столбцам."""
    labels = list(oof)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for r in range(2, len(labels) + 1):
        for combo in itertools.combinations(labels, r):
            Z = np.column_stack([oof[l] for l in combo])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm = cross_val_predict(_meta(), Z, y, cv=skf, method="predict_proba")[:, 1]
            rows.append({"комбинация": " + ".join(combo), "размер": r,
                         "ROC_AUC": round(roc_auc_score(y, pm), 4)})
    return pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)


def _stack_estimator(combo) -> StackingClassifier:
    """Настоящий StackingClassifier из тюнингованных базовых пайплайнов.

    Каждый базовый пайплайн сам выбирает свои признаки по именам столбцов, поэтому
    подаем общий фрейм признаков (надмножество). Мета-модель - логрег.
    """
    df = io.load_processed()
    _, _, y_train, _ = features.make_split(df)
    params = _load_params()
    estimators = []
    for lab in combo:
        model, fset = lab.split("|")
        feats = features.feature_sets(df)[fset]
        estimators.append((lab.replace("|", "_"),
                           ot._build(model, feats, df, y_train, params[lab])))
    return StackingClassifier(estimators=estimators, final_estimator=_meta(),
                              stack_method="predict_proba", cv=5)


def evaluate_on_test(combo) -> dict:
    """Обучает лучший стек на train, оценивает ROC-AUC на отложенном тесте."""
    df = io.load_processed()
    X_train, X_test, y_train, y_test = features.make_split(df)
    # Общий фрейм признаков - надмножество (no_collinear включает significant).
    sup = features.feature_sets(df)["no_collinear"]
    stack = _stack_estimator(combo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack.fit(X_train[sup], y_train)
        proba = stack.predict_proba(X_test[sup])[:, 1]
    return {"комбинация": " + ".join(combo),
            "test_ROC_AUC": round(roc_auc_score(y_test, proba), 3)}


def run() -> pd.DataFrame:
    oof, single, y = base_oof()
    table = screen_combinations(oof, y)
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "stacking_combinations.csv", index=False,
                 encoding="utf-8-sig")
    print("\nТоп-5 комбинаций по OOF ROC-AUC:")
    print(table.head(5).to_string(index=False))
    print(f"\nЛучшая одиночная: {max(single, key=single.get)} = {max(single.values()):.3f}")
    return table


if __name__ == "__main__":
    run()
