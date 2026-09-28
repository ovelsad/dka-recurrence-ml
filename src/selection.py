"""Отбор признаков для логрега (шаг 7)

Сначала C подбирается по сетке на всех признаках с L2 по log loss
(models.logreg_search, шаг 6). С этими параметрами признаки отбираются пятью
методами. Отбираем исходные признаки: номинальный признак входит или выходит
всеми уровнями сразу. Качество набора - ROC-AUC внутренней CV на обучающей
части (models.inner_folds).

Методы:
- одномерный: признаки упорядочены по ROC-AUC логрега на одном признаке и
  добавляются по этому порядку;
- жадный вперед: на каждом шаге добавляется признак, с которым ROC-AUC выше
  всего;
- жадный назад: начинаем со всех признаков и на каждом шаге убираем признак,
  без которого ROC-AUC выше всего;
- L1: логрег с L1-штрафом на всех признаках, C подбирается внутренней CV,
  отобраны признаки с ненулевыми коэффициентами;
- elastic net: то же, подбираются C и доля L1.

У первых трех методов остановка одна: если PATIENCE шагов подряд ROC-AUC не
превысил лучший, поиск заканчивается, берется набор с лучшим значением.
Итоговый набор "голосование" - признаки, которые отобрали не меньше VOTES
методов из пяти. Модель любого набора - логрег с параметрами сетки
"""

from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from . import evaluate, models
from .importance import original_feature

METHODS = ["одномерный", "жадный вперед", "жадный назад", "L1", "elastic net"]
PATIENCE = 2  # шагов подряд без улучшения до остановки
VOTES = 3  # признак в итоговом наборе, если его отобрали не меньше 3 методов из 5
ALL = "все признаки"
VOTE = "голосование"


def model_params(search: GridSearchCV) -> dict:
    """Параметры LogisticRegression из подбора по сетке, без префикса model__"""
    return {
        key.removeprefix("model__"): value for key, value in search.best_params_.items()
    }


def inner_auc(features: list, X: pd.DataFrame, y: pd.Series, params: dict) -> float:
    """ROC-AUC внутренней CV логрега с параметрами params на признаках features"""
    scores = cross_val_score(
        models.logreg(features, **params),
        X,
        y,
        cv=models.inner_folds(),
        scoring="roc_auc",
    )
    return scores.mean()


def add_until_stall(X, y, params: dict, order: list | None = None) -> list:
    """Добавление признаков по одному до остановки

    order - готовый порядок (одномерный метод). Без него на каждом шаге
    выбирается признак, с которым ROC-AUC выше всего (жадный вперед)
    """
    selected, remaining = [], list(order or X.columns)
    best_score, best_set, stall = -np.inf, [], 0
    while remaining and stall < PATIENCE:
        if order is None:
            scores = {f: inner_auc([*selected, f], X, y, params) for f in remaining}
            feature = max(scores, key=scores.get)
            score = scores[feature]
        else:
            feature = remaining[0]
            score = inner_auc([*selected, feature], X, y, params)
        selected.append(feature)
        remaining.remove(feature)
        if score > best_score:
            best_score, best_set, stall = score, list(selected), 0
        else:
            stall += 1
    return best_set


def remove_until_stall(X, y, params: dict) -> list:
    """Жадный назад: убираем по одному признаку до остановки"""
    selected = list(X.columns)
    best_score, best_set, stall = inner_auc(selected, X, y, params), list(selected), 0
    while len(selected) > 1 and stall < PATIENCE:
        scores = {
            f: inner_auc([g for g in selected if g != f], X, y, params)
            for f in selected
        }
        feature = max(scores, key=scores.get)  # его удаление дает лучший ROC-AUC
        selected.remove(feature)
        if scores[feature] > best_score:
            best_score, best_set, stall = scores[feature], list(selected), 0
        else:
            stall += 1
    return best_set


def univariate_order(X, y, params: dict) -> list:
    """Признаки по убыванию ROC-AUC внутренней CV логрега на одном признаке"""
    scores = {f: inner_auc([f], X, y, params) for f in X.columns}
    return sorted(scores, key=scores.get, reverse=True)


def penalized(X, y, penalty: str) -> list:
    """Признаки с ненулевыми коэффициентами логрега с L1 или elastic net

    C (и доля L1 у elastic net) подбираются внутренней CV по log loss, как в
    сетке шага 6
    """
    grid = {
        "model__C": models.C_GRID,
        "model__penalty": [penalty],
        "model__solver": ["saga"],
    }
    if penalty == "elasticnet":
        grid["model__l1_ratio"] = models.L1_RATIOS
    search = GridSearchCV(
        models.logreg(), grid, scoring=models.C_SCORING, cv=models.inner_folds()
    ).fit(X, y)
    fitted = search.best_estimator_
    columns = fitted[:-1].get_feature_names_out()
    coef = fitted.named_steps["model"].coef_[0]
    kept = {original_feature(c) for c, value in zip(columns, coef) if value != 0}
    return [f for f in X.columns if f in kept]


def select(method: str, X, y, params: dict) -> list:
    """Набор признаков, отобранный методом method на X, y"""
    if method == "одномерный":
        return add_until_stall(X, y, params, univariate_order(X, y, params))
    if method == "жадный вперед":
        return add_until_stall(X, y, params)
    if method == "жадный назад":
        return remove_until_stall(X, y, params)
    if method == "L1":
        return penalized(X, y, "l1")
    if method == "elastic net":
        return penalized(X, y, "elasticnet")
    raise ValueError(f"Неизвестный метод отбора {method!r}")


def vote(sets: dict, features: list) -> list:
    """Признаки, которые отобрали не меньше VOTES методов, в порядке features"""
    counts = Counter(f for chosen in sets.values() for f in chosen)
    return [f for f in features if counts[f] >= VOTES]


def all_sets(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    """Подбор по сетке на всех признаках, затем наборы всех методов и голосование

    Возвращает параметры логрега и {название набора: признаки}
    """
    params = model_params(models.logreg_search(n_jobs=1).fit(X, y))
    sets = {method: select(method, X, y, params) for method in METHODS}
    sets[VOTE] = vote(sets, list(X.columns))
    return params, sets


class SelectedLogreg(ClassifierMixin, BaseEstimator):
    """Логрег на наборе признаков, отобранном методом method

    method - один из METHODS или VOTE. При fit параметры подбираются по сетке
    на всех признаках, затем отбираются признаки и обучается модель. После fit:
    params_ - параметры логрега, sets_ - наборы методов, features_ - признаки
    модели
    """

    def __init__(self, method: str = VOTE):
        self.method = method

    def fit(self, X, y):
        """Подбор, отбор и обучение на X, y"""
        self.params_, self.sets_ = all_sets(X, y)
        self.features_ = self.sets_[self.method]
        if not self.features_:
            raise ValueError(f"Метод {self.method!r} не отобрал ни одного признака")
        self.model_ = models.logreg(self.features_, **self.params_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        """Вероятности классов модели на отобранных признаках"""
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Классы при пороге по умолчанию (0,5)"""
        return self.model_.predict(X)


# ---------------------------------------------------------------- вложенная оценка


def fold_run(X, y, train_idx, valid_idx) -> tuple[dict, dict, dict]:
    """Один внешний фолд: наборы всех методов на обучающей части и прогнозы

    Возвращает параметры логрега, наборы и прогнозы для валидационной части
    """
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    params, sets = all_sets(X_train, y_train)
    predictions = {
        name: models.logreg(features, **params)
        .fit(X_train, y_train)
        .predict_proba(X.iloc[valid_idx])[:, 1]
        for name, features in sets.items()
        if features
    }
    return params, sets, predictions


def cross_validate_methods(X: pd.DataFrame, y: pd.Series) -> list:
    """Вложенная оценка всех методов отбора и голосования за один проход по фолдам

    В каждом внешнем фолде на обучающей части подбирается сетка, отбираются
    признаки всеми методами, модели проверяются на валидационной части. Фолды
    считаются параллельно. Результаты в формате evaluate.cross_validate, в
    параметрах по фолдам - признаки и параметры логрега
    """
    if not X.index.equals(y.index):
        raise ValueError("Индексы X и y не совпадают")
    splits = list(evaluate.outer_folds().split(X, y))
    runs = Parallel(n_jobs=-1)(
        delayed(fold_run)(X, y, train_idx, valid_idx) for train_idx, valid_idx in splits
    )
    results = []
    for name in [*METHODS, VOTE]:
        oof = pd.DataFrame(np.nan, index=y.index, columns=range(evaluate.N_REPEATS))
        fold_aucs, fold_params = [], []
        for fold, ((_, valid_idx), (params, sets, predictions)) in enumerate(
            zip(splits, runs)
        ):
            if name not in predictions:
                raise ValueError(f"Метод {name!r} не отобрал признаков в фолде {fold}")
            oof.iloc[valid_idx, fold // evaluate.N_FOLDS] = predictions[name]
            fold_aucs.append(roc_auc_score(y.iloc[valid_idx], predictions[name]))
            fold_params.append({"признаки": sets[name], **params})
        results.append(
            {
                "модель": f"логрег, {name}",
                "ROC-AUC по фолдам": fold_aucs,
                "параметры по фолдам": fold_params,
                "пробы по фолдам": [None] * len(splits),
                "oof": oof,
                "y": y,
            }
        )
    return results


def selection_frequency(results: list, features: list) -> pd.DataFrame:
    """Доля внешних фолдов, в которых признак отобран: признаки x методы"""
    table = {}
    for result in results:
        counts = Counter(
            f for p in result["параметры по фолдам"] for f in p["признаки"]
        )
        table[result["модель"].removeprefix("логрег, ")] = [
            counts[f] / len(result["параметры по фолдам"]) for f in features
        ]
    return pd.DataFrame(table, index=features)


def agreement(sets: dict, features: list) -> pd.DataFrame:
    """Какие методы отобрали признак на всем train: 1 - отобран, и число голосов"""
    table = pd.DataFrame(
        {name: [int(f in chosen) for f in features] for name, chosen in sets.items()},
        index=features,
    )
    table["голосов"] = table[METHODS].sum(axis=1)
    return table.sort_values("голосов", ascending=False)
