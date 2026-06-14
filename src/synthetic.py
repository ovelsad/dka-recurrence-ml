"""Влияние синтетической балансировки на качество (этап 12).

Проверяем честно, дает ли порождение синтетических примеров меньшего класса
(рецидивов) прирост качества. Берем лучшую не-CatBoost модель из отложенного теста
- случайный лес на наборе no_collinear с тюнингованными гиперпараметрами (ноутбук
07). CatBoost не берем: он работает с категориями нативно, а методы SMOTE требуют
числовой кодировки.

Сравниваем стратегии:
- none: без балансировки (class_weight отключен), базовая линия;
- class_weight: взвешивание классов без синтетики, ориентир;
- smotenc: SMOTENC, корректный метод для смешанных данных (категориальные не
  интерполируются, а выбираются из соседей);
- borderline: BorderlineSMOTE (синтетика у границы классов);
- adasyn: ADASYN (больше синтетики там, где классы трудно разделимы);
- smote_onehot: обычный SMOTE поверх one-hot (для категорий некорректно, оставлен
  чтобы показать разницу с SMOTENC).

Честность оценки: оверсэмплинг выполняется только на train внутри фолдов, оценка
всегда на реальной (не синтетической) отложенной части фолда. Считаем несколько
seed-ов (разные разбиения и инициализация), приводим среднее и разброс. Главные
метрики - ROC-AUC (ранжирование), PR-AUC (важна при дисбалансе) и Brier
(калибровка): синтетика двигает баланс классов и порог, а не ранжирование, и часто
ухудшает калибровку, поэтому смотрим не только на AUC.
"""

import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config, features, io
from .config import RANDOM_SEED

STRATEGIES = ["none", "class_weight", "smotenc", "borderline", "adasyn", "smote_onehot"]
SEEDS = [RANDOM_SEED, 1, 2, 3, 4]
FSET = "no_collinear"


def _rf_params():
    """Тюнингованные параметры rf на no_collinear без class_weight (его задаем сами)."""
    params = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8")
    )["rf|no_collinear"].copy()
    params.pop("class_weight", None)
    return params


def _fold_metrics(yva, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(yva, proba),
        "pr_auc": average_precision_score(yva, proba),
        "sens": tp / (tp + fn) if (tp + fn) else np.nan,
        "spec": tn / (tn + fp) if (tn + fp) else np.nan,
        "brier": brier_score_loss(yva, proba),
    }


def _resample(strategy, X_enc, X_mixed, y, n_num, ord_enc, onehot, seed):
    """Обучающая матрица (в кодировке модели) после оверсэмплинга."""
    if strategy in ("none", "class_weight"):
        return X_enc, y
    if strategy == "smote_onehot":
        return SMOTE(random_state=seed).fit_resample(X_enc, y)
    if strategy == "borderline":
        return BorderlineSMOTE(random_state=seed).fit_resample(X_enc, y)
    if strategy == "adasyn":
        return ADASYN(random_state=seed).fit_resample(X_enc, y)
    if strategy == "smotenc":
        cat_idx = list(range(n_num, X_mixed.shape[1]))
        sm = SMOTENC(categorical_features=cat_idx, random_state=seed)
        mixed_res, y_res = sm.fit_resample(X_mixed, y)
        num_part = mixed_res[:, :n_num]
        cat_part = ord_enc.inverse_transform(mixed_res[:, n_num:])
        enc = np.hstack([num_part, onehot.transform(cat_part)])
        return enc, y_res
    raise ValueError(strategy)


def _evaluate_seed(strategy: str, seed: int, rf_params: dict) -> dict:
    """Средние по фолдам метрики для стратегии при одном seed."""
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[FSET]
    quant, cat = features.column_types(df, feats)

    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = {k: [] for k in ["roc_auc", "pr_auc", "sens", "spec", "brier"]}

    for tr, va in skf.split(X_train, y_train):
        Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
        ytr, yva = y_train.iloc[tr], y_train.iloc[va]

        num_imp = KNNImputer(n_neighbors=5).fit(Xtr[quant])
        scaler = StandardScaler().fit(num_imp.transform(Xtr[quant]))
        ntr = scaler.transform(num_imp.transform(Xtr[quant]))
        nva = scaler.transform(num_imp.transform(Xva[quant]))

        cat_imp = SimpleImputer(strategy="most_frequent").fit(Xtr[cat].astype(object))
        ctr = cat_imp.transform(Xtr[cat].astype(object))
        cva = cat_imp.transform(Xva[cat].astype(object))

        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(ctr)
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit(ctr)

        Xtr_enc = np.hstack([ntr, onehot.transform(ctr)])
        Xva_enc = np.hstack([nva, onehot.transform(cva)])
        Xtr_mixed = np.hstack([ntr, ord_enc.transform(ctr)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Xres, yres = _resample(strategy, Xtr_enc, Xtr_mixed, ytr,
                                       ntr.shape[1], ord_enc, onehot, seed)
            except Exception:  # ADASYN иногда не может породить примеры
                return {k: np.nan for k in folds}
            cw = "balanced" if strategy == "class_weight" else None
            clf = RandomForestClassifier(**rf_params, class_weight=cw,
                                         random_state=seed, n_jobs=-1).fit(Xres, yres)
            proba = clf.predict_proba(Xva_enc)[:, 1]
        for k, v in _fold_metrics(yva, proba).items():
            folds[k].append(v)

    return {k: float(np.nanmean(v)) for k, v in folds.items()}


def evaluate(strategy: str, rf_params: dict) -> dict:
    """Среднее и SD метрик по нескольким seed-ам."""
    per_seed = [_evaluate_seed(strategy, s, rf_params) for s in SEEDS]
    out = {"стратегия": strategy}
    for k in ["roc_auc", "pr_auc", "sens", "spec", "brier"]:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        out[k] = round(float(np.nanmean(vals)), 3)
        out[f"{k}_sd"] = round(float(np.nanstd(vals)), 3)
    return out


def run() -> pd.DataFrame:
    rf_params = _rf_params()
    rows = []
    for s in STRATEGIES:
        res = evaluate(s, rf_params)
        rows.append(res)
        print(f"{s:14} ROC-AUC={res['roc_auc']}±{res['roc_auc_sd']} "
              f"PR-AUC={res['pr_auc']}±{res['pr_auc_sd']} Brier={res['brier']}")
    table = pd.DataFrame(rows)
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "synthetic.csv", index=False, encoding="utf-8-sig")
    return table


if __name__ == "__main__":
    table = run()
    print()
    print(table.to_string(index=False))
