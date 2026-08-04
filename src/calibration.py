"""Калибровка, выбор порога и конфигурация финалистов (этап 9, автоматизация).

Автоматизация ноутбука 09. Считает out-of-fold предсказания финалистов на train,
сравнивает калибровку (сырые вероятности, Платт, изотоническая), подбирает пороги
и фиксирует по каждому финалисту калибровку и порог под целевую чувствительность.
Все out-of-fold, без утечки: порог берется с train, тест трогается один раз.

Пишет: calibration_oof.csv, threshold_train_oof.csv, threshold_test.csv,
finalists_config.json. Гиперпараметры берет из tuning_optuna_params.json (этап 7).
"""

import json
import warnings

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io, threshold as thr
from . import optuna_tuning as ot
from .config import RANDOM_SEED

# Финалисты: четыре модели без LightGBM (он не проходил в шортлист финалистов).
MODELS = ["logreg", "rf", "xgb", "catboost"]
SETS = ["significant", "no_collinear"]
FINALISTS = [(m, fs) for fs in SETS for m in MODELS]

SENS_TARGET = 0.75      # целевая чувствительность для рабочего порога
BRIER_MARGIN = 0.005    # насколько Платт должен улучшить Brier, чтобы его брать


def _params():
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def run():
    df = io.load_processed()
    X_train, X_test, y_train, y_test = features.make_split(df)
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()
    params_all = _params()
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

    def build(model, fset):
        feats = features.feature_sets(df)[fset]
        pipe = ot._build(model, feats, df, y_train, params_all[f"{model}|{fset}"])
        return pipe, feats

    # 1. Out-of-fold вероятности на train (честная оценка, без утечки).
    oof = {}
    for model, fset in FINALISTS:
        pipe, feats = build(model, fset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = cross_val_predict(pipe, X_train[feats], y_train, cv=skf,
                                      method="predict_proba")[:, 1]
        oof[(model, fset)] = proba
        print(f"{model:9} {fset:13} OOF ROC-AUC={roc_auc_score(y_tr, proba):.3f}",
              flush=True)

    # 2. Калибровка: сырые, Платт, изотоническая (все out-of-fold).
    calib_oof, calib_rows = {}, []
    for model, fset in FINALISTS:
        pipe, feats = build(model, fset)
        variants = {"сырые": oof[(model, fset)]}
        for method, label in [("sigmoid", "платт"), ("isotonic", "изотон.")]:
            cc = CalibratedClassifierCV(pipe, method=method, cv=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                variants[label] = cross_val_predict(cc, X_train[feats], y_train,
                                                    cv=skf, method="predict_proba")[:, 1]
        calib_oof[(model, fset)] = variants
        for label, proba in variants.items():
            calib_rows.append({"модель": model, "набор": fset, "калибровка": label,
                               "Brier": round(brier_score_loss(y_tr, proba), 3),
                               "ROC-AUC": round(roc_auc_score(y_tr, proba), 3)})
    pd.DataFrame(calib_rows).to_csv(config.TABLES_DIR / "calibration_oof.csv",
                                    index=False, encoding="utf-8-sig")

    # 3. Пороги на train OOF: Youden, sens>=0.75, sens>=0.80.
    rows = []
    for model, fset in FINALISTS:
        proba = oof[(model, fset)]
        ch = thr.select_thresholds(y_tr, proba, sens_target=0.75)
        ch80 = thr.select_thresholds(y_tr, proba, sens_target=0.80)
        for label, t in [("Youden", ch["Youden"]),
                         ("sens>=0.75", ch["чувств.>=0.75"]),
                         ("sens>=0.80", ch80["чувств.>=0.8"])]:
            rows.append({"модель": model, "набор": fset, "критерий": label,
                         **thr.metrics_at(y_tr, proba, t)})
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "threshold_train_oof.csv",
                              index=False, encoding="utf-8-sig")

    # 4. Выбор калибровки и порога, проверка на тесте, конфигурация финалистов.
    rows, finalists_cfg, conf_rows = [], {}, []
    for model, fset in FINALISTS:
        pipe, feats = build(model, fset)
        b_raw = brier_score_loss(y_tr, calib_oof[(model, fset)]["сырые"])
        b_platt = brier_score_loss(y_tr, calib_oof[(model, fset)]["платт"])
        use_platt = b_platt <= b_raw - BRIER_MARGIN

        oof_use = calib_oof[(model, fset)]["платт"] if use_platt else oof[(model, fset)]
        sel = thr.select_thresholds(y_tr, oof_use, sens_target=SENS_TARGET)
        t = sel[f"чувств.>={SENS_TARGET}"]
        calib_name = "платт" if use_platt else "сырые"

        est = CalibratedClassifierCV(pipe, method="sigmoid", cv=5) if use_platt else pipe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train[feats], y_train)
            proba_test = est.predict_proba(X_test[feats])[:, 1]

        m_tr = thr.metrics_at(y_tr, oof_use, t)
        m_te = thr.metrics_at(y_te, proba_test, t)
        rows.append({"модель": model, "набор": fset,
                     "калибровка": calib_name,
                     "порог": round(float(t), 3),
                     "sens train": m_tr["чувств."], "spec train": m_tr["специф."],
                     "sens test": m_te["чувств."], "spec test": m_te["специф."],
                     "PPV test": m_te["PPV"], "NPV test": m_te["NPV"]})
        finalists_cfg[f"{model}|{fset}"] = {
            "feature_set": fset,
            "calibration": "sigmoid" if use_platt else "none",
            "threshold": round(float(t), 4),
            "sens_target": SENS_TARGET}

        # Матрица ошибок на тесте: сырые счета TP/FP/FN/TN при двух порогах,
        # оба подобраны на train OOF (Юден и целевая чувствительность).
        for crit, t_c in [("Юден", sel["Youden"]), (f"sens>={SENS_TARGET}", t)]:
            pred = (proba_test >= t_c).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            mc = thr.metrics_at(y_te, proba_test, t_c)
            conf_rows.append({"модель": model, "набор": fset, "калибровка": calib_name,
                              "критерий": crit, "порог": round(float(t_c), 3),
                              "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                              "sens": mc["чувств."], "spec": mc["специф."],
                              "PPV": mc["PPV"], "NPV": mc["NPV"]})

    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "threshold_test.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(conf_rows).to_csv(config.TABLES_DIR / "confusion_youden.csv",
                                   index=False, encoding="utf-8-sig")
    (config.TABLES_DIR / "finalists_config.json").write_text(
        json.dumps(finalists_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Калибровка, пороги и матрица ошибок готовы, финалисты записаны.", flush=True)


if __name__ == "__main__":
    run()
