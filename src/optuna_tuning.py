"""Подбор гиперпараметров через Optuna с вложенной кросс-валидацией (этап 7).

Optuna ведет поиск TPE-сэмплером: строит вероятностную модель связи
гиперпараметров и качества и направляет следующие пробы в перспективные области,
а не перебирает вслепую, как случайный или полный поиск. Прунинг (MedianPruner)
досрочно отсекает заведомо слабые пробы по промежуточным фолдам.

Честная оценка - вложенная CV: внешние фолды оценивают обобщение, внутри каждого
Optuna подбирает гиперпараметры по внутренней CV. Отдельное финальное исследование
на всем train дает гиперпараметры для развертывания и оптимистичную внутреннюю
оценку.

Зафиксировано по ноутбуку выбора стратегий (06): импутация KNN, кодирование one-hot
(CatBoost - нативные категории). Балансировка входит в пространство поиска через
class_weight / scale_pos_weight / auto_class_weights. Метрика - ROC-AUC.
"""

import warnings

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import features, io
from .config import RANDOM_SEED
from .modeling import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

SHORTLIST = ["logreg", "rf", "lgbm", "catboost", "xgb"]
FSETS = ["significant", "no_collinear"]
IMPUTATION = "knn"


def suggest_params(trial, model: str) -> dict:
    """Пространство поиска гиперпараметров для каждой модели."""
    if model == "logreg":
        # Перебираем тип регуляризации: L1 (отбор признаков), L2 (сжатие),
        # elasticnet (смесь, доля L1 задается l1_ratio). Решатель под штраф
        # подбирается детерминированно в _build (результат от решателя не зависит).
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": penalty,
            "class_weight": trial.suggest_categorical("class_weight",
                                                      [None, "balanced"]),
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features",
                                                      ["sqrt", "log2", 0.5, 0.8]),
            "criterion": trial.suggest_categorical("criterion",
                                                   ["gini", "entropy", "log_loss"]),
            "class_weight": trial.suggest_categorical("class_weight",
                                                      [None, "balanced"]),
        }
    if model == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": trial.suggest_categorical("class_weight",
                                                      [None, "balanced"]),
        }
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }
    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 800, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights",
                                                            [None, "Balanced"]),
        }
    raise ValueError(model)


def _build(model, feats, df, y, params):
    """Пайплайн с рабочими стратегиями (KNN, one-hot/native) и заданными
    гиперпараметрами классификатора."""
    pipe = build_pipeline(df, feats, model, IMPUTATION, "none", y)
    # None не передаем в set_params: базовый классификатор и так создан с None
    # по умолчанию (balancing="none"), а CatBoost не парсит auto_class_weights=None.
    setp = {f"clf__{k}": v for k, v in params.items() if v is not None}
    if model == "logreg":
        # Решатель под штраф: результат один и тот же (сходятся к одному оптимуму),
        # разница только в скорости. lbfgs для l2, liblinear для l1, saga для
        # elasticnet (единственный поддерживает смесь).
        setp["clf__solver"] = {"l2": "lbfgs", "l1": "liblinear",
                               "elasticnet": "saga"}[params.get("penalty", "l2")]
    pipe.set_params(**setp)
    return pipe


def _cv_score(pipe, X, y, cv, trial=None) -> float:
    """Средний ROC-AUC по фолдам с возможностью прунинга по ходу."""
    scores = []
    for i, (tr, va) in enumerate(cv.split(X, y)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X.iloc[tr], y.iloc[tr])
            proba = pipe.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], proba))
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def _make_objective(model, feats, df, X, y, inner_cv):
    def objective(trial):
        params = suggest_params(trial, model)
        pipe = _build(model, feats, df, y, params)
        return _cv_score(pipe, X, y, inner_cv, trial)
    return objective


def _study(seed):
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))


def nested_cv(model: str, fset: str, n_trials: int = 40, seed: int = RANDOM_SEED,
              df=None):
    """Вложенная CV: внешние 5 фолдов для честной оценки, внутри Optuna (3 фолда).

    df - обработанный датасет; если None, берется текущий io.load_processed().
    Явная передача df позволяет прогнать тюнинг на другой базе (старой/новой) без
    подмены рабочих файлов.
    """
    if df is None:
        df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)

    scores = []
    for tr, va in outer.split(X_train, y_train):
        Xtr, ytr = X_train.iloc[tr][feats], y_train.iloc[tr]
        Xva, yva = X_train.iloc[va][feats], y_train.iloc[va]
        study = _study(seed)
        study.optimize(_make_objective(model, feats, df, Xtr, ytr, inner),
                       n_trials=n_trials)
        best = _build(model, feats, df, ytr, study.best_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best.fit(Xtr, ytr)
            proba = best.predict_proba(Xva)[:, 1]
        scores.append(roc_auc_score(yva, proba))
    return np.array(scores)


def final_study(model: str, fset: str, n_trials: int = 100, seed: int = RANDOM_SEED,
                df=None):
    """Финальное исследование на всем train (внутренняя 5-фолдовая CV)."""
    if df is None:
        df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    study = _study(seed)
    study.optimize(_make_objective(model, feats, df, X_train[feats], y_train, inner),
                   n_trials=n_trials)
    return study


def run_tuning(n_nested: int = 40, n_final: int = 100, df=None, fsets=None,
               out_dir=None):
    """Прогоняет тюнинг по моделям и наборам, пишет CSV и JSON параметров.

    Автоматизация ноутбука 07. Для каждой модели и набора считает вложенную CV
    (честная оценка обобщения) и финальное исследование (гиперпараметры для
    развертывания). Результаты пишутся по ходу, чтобы длинный прогон был устойчив.

    df - обработанный датасет (None -> текущий); fsets - список наборов признаков
    (None -> все FSETS); out_dir - куда писать (None -> config.TABLES_DIR). Через
    df/out_dir один и тот же тюнинг прогоняется на старой и новой базах в разные
    папки. Возвращает таблицу с итогами.
    """
    import json
    import time

    import pandas as pd

    from . import config

    if fsets is None:
        fsets = FSETS
    config.ensure_dirs()
    out_dir = config.TABLES_DIR if out_dir is None else out_dir

    rows, best_params = [], {}
    for fs in fsets:
        for m in SHORTLIST:
            t0 = time.time()
            nested = nested_cv(m, fs, n_trials=n_nested, df=df)
            study = final_study(m, fs, n_trials=n_final, df=df)
            best_params[f"{m}|{fs}"] = study.best_params
            rows.append({
                "модель": m, "набор": fs,
                "вложенная_ROC_AUC": round(float(nested.mean()), 3),
                "SD": round(float(nested.std()), 3),
                "внутренняя_ROC_AUC": round(float(study.best_value), 3),
                "лучшие_параметры": study.best_params,
            })
            print(f"{m:9} {fs:13} вложенная={nested.mean():.3f}+-{nested.std():.3f} "
                  f"внутренняя={study.best_value:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "tuning_optuna.csv",
                                      index=False, encoding="utf-8-sig")
            (out_dir / "tuning_optuna_params.json").write_text(
                json.dumps(best_params, ensure_ascii=False, indent=2),
                encoding="utf-8")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_tuning()
