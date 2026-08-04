"""Сводная оценка на отложенном тесте (этап 8, автоматизация ноутбука 08).

Обучает каждую пару модель|набор из tuning_optuna_params.json на train, один раз
считает предсказания на отложенном тесте и собирает метрики с бутстрэп-ДИ.
Результат - таблица test_comparison.csv, отсортированная по ROC-AUC.

Порог и калибровку сюда не вносим, это делает src.calibration (этап 9). Здесь
чистая дискриминация на тесте по всем моделям шортлиста, включая LightGBM.
"""

import json
import warnings

import pandas as pd

from . import config, evaluation, features, io
from . import optuna_tuning as ot


def run():
    df = io.load_processed()
    X_train, X_test, y_train, y_test = features.make_split(df)
    params_all = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))
    print(f"train {len(y_train)}, test {len(y_test)}, моделей {len(params_all)}",
          flush=True)

    rows = []
    for key, params in params_all.items():
        model, fset = key.split("|")
        feats = features.feature_sets(df)[fset]
        pipe = ot._build(model, feats, df, y_train, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train[feats], y_train)
            proba = pipe.predict_proba(X_test[feats])[:, 1]
        m = evaluation.bootstrap_metrics(y_test, proba)
        rows.append({
            "модель": model, "набор": fset,
            **{k: f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]" for k, v in m.items()},
            "_auc": m["ROC-AUC"][0]})
        print(f"{model:9} {fset:13} ROC-AUC={m['ROC-AUC'][0]:.3f}", flush=True)

    comp = (pd.DataFrame(rows)
            .sort_values("_auc", ascending=False)
            .drop(columns="_auc").reset_index(drop=True))
    config.ensure_dirs()
    comp.to_csv(config.TABLES_DIR / "test_comparison.csv",
                index=False, encoding="utf-8-sig")
    print("Сводка по тесту записана в test_comparison.csv", flush=True)
    return comp


if __name__ == "__main__":
    run()
