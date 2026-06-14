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
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight",
                                                      [None, "balanced"]),
        }
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features",
                                                      ["sqrt", "log2", 0.5, 0.8]),
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
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": trial.suggest_categorical("class_weight",
                                                      [None, "balanced"]),
        }
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        }
    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 800, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
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
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items() if v is not None})
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


def nested_cv(model: str, fset: str, n_trials: int = 40, seed: int = RANDOM_SEED):
    """Вложенная CV: внешние 5 фолдов для честной оценки, внутри Optuna (3 фолда)."""
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


def final_study(model: str, fset: str, n_trials: int = 100, seed: int = RANDOM_SEED):
    """Финальное исследование на всем train (внутренняя 5-фолдовая CV)."""
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    study = _study(seed)
    study.optimize(_make_objective(model, feats, df, X_train[feats], y_train, inner),
                   n_trials=n_trials)
    return study
