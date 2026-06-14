"""Бутстрэп-доверительные интервалы и OOB-оценка финалистов (этап 13).

Два независимых подтверждения устойчивости качества при малой выборке, на
тюнингованных финалистах (логрег, лес, XGBoost, CatBoost на наборах significant и
no_collinear), гиперпараметры из ноутбука 07.

1. Бутстрэп-ДИ. Для каждого финалиста считаем ROC-AUC и PR-AUC на out-of-fold
   предсказаниях train (без утечки) с 95% доверительным интервалом по бутстрэпу
   (2000 повторов). ДИ доступен для любой модели, он не связан с OOB.

2. OOB-AUC. Для случайного леса (каноничный бэггинг деревьев): оценка по
   наблюдениям, не попавшим в бутстрэп-подвыборку дерева. У бустинга (XGBoost,
   CatBoost) OOB в этом смысле нет: деревья строятся последовательно, бутстрэп-мешка
   нет, переобучение контролируют регуляризацией и числом итераций. У логрега OOB
   тоже нет. Бэггинг логрегов не берем: бэггинг помогает моделям с высокой
   дисперсией (деревья), у логрега она низкая.

Оговорка: предобработка (импутация, масштабирование, кодирование) при расчете OOB
обучается на всем train, поэтому OOB-оценка слегка оптимистична. Она служит вторичной
проверкой согласия с out-of-fold, а не самостоятельной несмещенной метрикой.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io
from . import optuna_tuning as ot
from .config import RANDOM_SEED
from .evaluation import bootstrap_metrics

MODELS = ["logreg", "rf", "xgb", "catboost"]
FSETS = ["significant", "no_collinear"]


def _params():
    import json
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def _oof_proba(df, X_train, y_train, model, fset, feats, params):
    """Out-of-fold вероятности тюнингованного финалиста на train (без утечки)."""
    pipe = ot._build(model, feats, df, y_train, params[f"{model}|{fset}"])
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                  method="predict_proba")[:, 1]
    return proba


def _oob_proba(df, X_train, y_train, fset, feats, params):
    """OOB-вероятности тюнингованного случайного леса."""
    pipe = ot._build("rf", feats, df, y_train, params[f"rf|{fset}"])
    pipe.set_params(clf__oob_score=True, clf__bootstrap=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train[feats], y_train)
    return pipe.named_steps["clf"].oob_decision_function_[:, 1]


def run() -> pd.DataFrame:
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    sets = features.feature_sets(df)
    params = _params()
    y = y_train.to_numpy()

    rows = []
    for fset in FSETS:
        for model in MODELS:
            feats = sets[fset]
            proba = _oof_proba(df, X_train, y_train, model, fset, feats, params)
            m = bootstrap_metrics(y, proba)
            roc, rlo, rhi = m["ROC-AUC"]
            pr, plo, phi = m["PR-AUC"]
            rows.append({"модель": model, "набор": fset, "n_призн": len(feats),
                         "ROC_AUC": roc, "ROC_AUC_ДИ": f"[{rlo}; {rhi}]",
                         "PR_AUC": pr, "PR_AUC_ДИ": f"[{plo}; {phi}]"})
            print(f"{model:9} {fset:13} ROC-AUC={roc} [{rlo}; {rhi}]")
    table = pd.DataFrame(rows)

    # OOB для случайного леса (каноничный бэггинг деревьев), оба набора.
    oob_rows = []
    for fset in FSETS:
        feats = sets[fset]
        proba = _oob_proba(df, X_train, y_train, fset, feats, params)
        m = bootstrap_metrics(y, proba)
        roc, rlo, rhi = m["ROC-AUC"]
        oob_rows.append({"модель": "rf", "набор": fset,
                         "OOB_ROC_AUC": roc, "OOB_ДИ": f"[{rlo}; {rhi}]"})
        print(f"OOB rf       {fset:13} ROC-AUC={roc} [{rlo}; {rhi}]")
    oob_table = pd.DataFrame(oob_rows)

    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "bootstrap_ci.csv", index=False,
                 encoding="utf-8-sig")
    oob_table.to_csv(config.TABLES_DIR / "oob_auc.csv", index=False,
                     encoding="utf-8-sig")

    lines = ["# Бутстрэп-ДИ и OOB-оценка (тюнингованные финалисты)", "",
             "OOF-предсказания на train, гиперпараметры из ноутбука 07. ДИ 95% по "
             "бутстрэпу (2000 повторов).", "",
             "## ROC-AUC и PR-AUC с 95% ДИ по финалистам", "",
             table.to_markdown(index=False), "",
             "## OOB-AUC (случайный лес, бэггинг деревьев)", "",
             "У случайного леса есть OOB-оценка по наблюдениям вне бутстрэп-мешка "
             "дерева. У бустинга и логрега OOB нет, для них устойчивость "
             "подтверждается бутстрэп-ДИ выше.", "",
             oob_table.to_markdown(index=False)]
    (config.REPORTS_DIR / "bootstrap_ci.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return table


if __name__ == "__main__":
    run()
