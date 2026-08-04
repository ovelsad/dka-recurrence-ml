"""Матрица ошибок и ошибки I/II рода на отложенном тесте БЕЗ калибровки и БЕЗ
подобранного порога.

Оценка ведется на сырых вероятностях модели при фиксированном пороге 0.5 - это
честная неподогнанная рабочая точка, одинаковая для любой базы. Плюс
пороговонезависимые метрики (ROC-AUC, PR-AUC) с бутстрэп-ДИ. Один и тот же метод
для новой (314) и старой (273) выборок дает сопоставимое сравнение.

Калибровку и подбор порога намеренно не трогаем: соответствующие этапы
(src/calibration.py) остаются в пайплайне на случай, когда понадобятся, но в это
сравнение не входят - калибровка не меняет ранжирование (ROC-AUC), а подобранный
на train порог плохо переносится на малый тест.

Ошибка I рода = FP/(FP+TN) = 1 - специфичность (ложная тревога).
Ошибка II рода = FN/(FN+TP) = 1 - чувствительность (пропуск рецидива).
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import config, evaluation, features, io
from . import optuna_tuning as ot

MODELS = ["logreg", "rf", "lgbm", "xgb", "catboost"]
FSET = "no_collinear"
THRESHOLD = 0.5


def evaluate(df, params, models=MODELS, fset=FSET, threshold=THRESHOLD) -> pd.DataFrame:
    """Матрица ошибок и метрики на тесте для каждой модели набора.

    df - обработанный датасет; params - словарь гиперпараметров вида "модель|набор".
    Модель обучается на train на сырых вероятностях, тест оценивается один раз.
    """
    X_train, X_test, y_train, y_test = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    y_te = y_test.to_numpy()
    rows = []
    for m in models:
        key = f"{m}|{fset}"
        if key not in params:
            continue
        pipe = ot._build(m, feats, df, y_train, params[key])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train[feats], y_train)
            proba = pipe.predict_proba(X_test[feats])[:, 1]

        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        err1 = fp / (fp + tn) if (fp + tn) else np.nan   # ложная тревога
        err2 = fn / (fn + tp) if (fn + tp) else np.nan   # пропуск рецидива
        b = evaluation.bootstrap_metrics(y_te, proba, threshold=threshold)
        rows.append({
            "модель": m, "набор": fset,
            "ROC_AUC": b["ROC-AUC"][0],
            "ROC_AUC_ДИ": f"[{b['ROC-AUC'][1]}; {b['ROC-AUC'][2]}]",
            "PR_AUC": b["PR-AUC"][0],
            "PR_AUC_ДИ": f"[{b['PR-AUC'][1]}; {b['PR-AUC'][2]}]",
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "ошибка_I_рода": round(float(err1), 3),
            "ошибка_II_рода": round(float(err2), 3),
            "чувствительность": round(tp / (tp + fn), 3) if (tp + fn) else np.nan,
            "специфичность": round(tn / (tn + fp), 3) if (tn + fp) else np.nan,
        })
    return (pd.DataFrame(rows)
            .sort_values("ROC_AUC", ascending=False).reset_index(drop=True))


def run():
    """Прогон на текущей обработанной базе и текущих гиперпараметрах."""
    df = io.load_processed()
    params = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))
    _, _, _, y_test = features.make_split(df)
    table = evaluate(df, params)
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "error_matrix.csv", index=False,
                 encoding="utf-8-sig")
    print(f"Тест: {len(y_test)} пациентов, порог {THRESHOLD}, сырые вероятности.")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
