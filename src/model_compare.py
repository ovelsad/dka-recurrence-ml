"""Попарное сравнение разделяющей способности финалистов (этап 15).

Заменяет узкую проверку "лучшая модель против логрега". Считаем попарные критерии
DeLong между всеми восемью финалистами отдельно на out-of-fold обучающей части и на
отложенном тесте, с поправкой Бенджамини-Хохберга на множественные сравнения. По числу
значимых пар судим, можно ли вообще выделить лучшую модель.

Отдельно проверяем методологическую гипотезу о сложности: каждое сложное семейство
(случайный лес, XGBoost, CatBoost) сравниваем с логистической регрессией на наборе без
мультиколлинеарности, на OOF и на тесте, критерием DeLong и парным бутстрэпом.

DeLong сравнивает связанные ROC-кривые (предсказания спарены по пациентам), метрика ROC
инвариантна к калибровке, поэтому берем сырые вероятности тюнингованных моделей.
"""

import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io, stats
from . import optuna_tuning as ot
from .config import RANDOM_SEED
from .hypothesis_test import delong_test, paired_bootstrap

MODELS = ["logreg", "rf", "xgb", "catboost"]
FSETS = ["significant", "no_collinear"]
LABELS = {"logreg": "логрег", "rf": "лес", "xgb": "XGBoost", "catboost": "CatBoost"}
SETLAB = {"significant": "10", "no_collinear": "25"}


def _params():
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def _fmt(key):
    model, fset = key
    return f"{LABELS[model]} ({SETLAB[fset]})"


def _oof_proba(df, X_train, y_train, model, fset, feats, params):
    """Out-of-fold вероятности тюнингованного финалиста на train (без утечки)."""
    pipe = ot._build(model, feats, df, y_train, params[f"{model}|{fset}"])
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                  method="predict_proba")[:, 1]
    return proba


def _test_proba(df, X_train, y_train, X_test, model, fset, feats, params):
    """Вероятности на отложенном тесте: обучаем на train, предсказываем test."""
    pipe = ot._build(model, feats, df, y_train, params[f"{model}|{fset}"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train[feats], y_train)
        return pipe.predict_proba(X_test[feats])[:, 1]


def _pairwise(keys, proba, y):
    """Матрица попарных DeLong по всем парам моделей с поправкой Бенджамини-Хохберга."""
    rows = []
    for a, b in itertools.combinations(keys, 2):
        auc_a, auc_b, z, p, _ = delong_test(y, proba[a], proba[b])
        rows.append({"модель A": _fmt(a), "модель B": _fmt(b),
                     "AUC A": round(auc_a, 3), "AUC B": round(auc_b, 3),
                     "разница": round(auc_a - auc_b, 3),
                     "z": round(z, 3), "p": round(p, 4)})
    table = pd.DataFrame(rows)
    table["p (BH)"] = np.round(stats.adjust_pvalues(table["p"].to_numpy()), 4)
    return table


def run() -> dict:
    df = io.load_processed()
    X_train, X_test, y_train, y_test = features.make_split(df)
    sets = features.feature_sets(df)
    params = _params()
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()

    oof, test = {}, {}
    keys = []
    for fset in FSETS:
        feats = sets[fset]
        for model in MODELS:
            key = (model, fset)
            keys.append(key)
            oof[key] = _oof_proba(df, X_train, y_train, model, fset, feats, params)
            test[key] = _test_proba(df, X_train, y_train, X_test, model, fset,
                                    feats, params)

    pair_oof = _pairwise(keys, oof, y_tr)
    pair_test = _pairwise(keys, test, y_te)
    n_sig_oof = int((pair_oof["p (BH)"] < 0.05).sum())
    n_sig_test = int((pair_test["p (BH)"] < 0.05).sum())

    print(f"OOF (n={len(y_tr)}): значимых пар после БХ {n_sig_oof} из {len(pair_oof)}")
    print(f"тест (n={len(y_te)}): значимых пар после БХ {n_sig_test} из {len(pair_test)}")

    # методологическая гипотеза: сложные против логрега на наборе без
    # мультиколлинеарности (несмещенная оценка, significant несет преселекцию).
    # 3 сложных семейства, поправка БХ по 3 на каждую выборку.
    fset = "no_collinear"
    logreg_key = ("logreg", fset)
    h2_rows = []
    for split, proba, y in [("OOF", oof, y_tr), ("тест", test, y_te)]:
        sub = []
        for model in ["rf", "xgb", "catboost"]:
            ck = (model, fset)
            auc_c, auc_s, z, p, _ = delong_test(y, proba[ck], proba[logreg_key])
            d_mean, d_lo, d_hi, p_boot = paired_bootstrap(y, proba[ck],
                                                          proba[logreg_key])
            sub.append({"выборка": split, "сложная": LABELS[model],
                        "AUC сложная": round(auc_c, 3), "AUC логрег": round(auc_s, 3),
                        "разница": round(auc_c - auc_s, 3),
                        "DeLong p": round(p, 4),
                        "бутстрэп ДИ": f"[{d_lo:.3f}; {d_hi:.3f}]"})
        sub = pd.DataFrame(sub)
        sub["DeLong p (BH)"] = np.round(stats.adjust_pvalues(sub["DeLong p"].to_numpy()), 4)
        h2_rows.append(sub)
    h2 = pd.concat(h2_rows, ignore_index=True)
    print("\nГипотеза о сложности (сложные против логрега, no_collinear):")
    print(h2.to_string(index=False))

    config.ensure_dirs()
    pair_oof.to_csv(config.TABLES_DIR / "pairwise_delong_oof.csv", index=False,
                    encoding="utf-8-sig")
    pair_test.to_csv(config.TABLES_DIR / "pairwise_delong_test.csv", index=False,
                     encoding="utf-8-sig")
    h2.to_csv(config.TABLES_DIR / "hypothesis_complexity.csv", index=False,
              encoding="utf-8-sig")

    lines = ["# Попарное сравнение финалистов критерием DeLong", "",
             f"OOF обучающей части (n={len(y_tr)}) и отложенный тест (n={len(y_te)}), "
             "поправка Бенджамини-Хохберга.", "",
             f"Значимых пар после поправки: OOF {n_sig_oof} из {len(pair_oof)}, "
             f"тест {n_sig_test} из {len(pair_test)}.", "",
             "## Попарный DeLong на OOF", "", pair_oof.to_markdown(index=False), "",
             "## Попарный DeLong на тесте", "", pair_test.to_markdown(index=False), "",
             "## Гипотеза о сложности: сложные против логрега (no_collinear)", "",
             h2.to_markdown(index=False)]
    (config.REPORTS_DIR / "model_compare.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    return {"n_sig_oof": n_sig_oof, "n_sig_test": n_sig_test,
            "pairs": len(pair_oof)}


if __name__ == "__main__":
    run()
