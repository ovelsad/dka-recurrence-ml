"""Совместные эффекты признаков и правила для врачей (шаг 10)

Взаимодействия SHAP. У древесной модели вклад каждого признака в прогноз
пациента раскладывается на вклад признака самого по себе и вклады пар с
другими признаками. Средний модуль вклада пары показывает, какие признаки
действуют вместе сильнее, чем сумма их вкладов по одному. У модели из пней
(дерево глубины 1) взаимодействий нет по построению.

Правила. Неглубокое дерево решений на всех признаках: каждый лист читается как
"если ... и ..., то рецидив у такой-то доли пациентов". Признаки идут как
есть, без заполнения и масштаба: пороги в единицах признака, пропуск дерево
обрабатывает само. Номинальные идут кодом. Глубина, минимальный лист, обрезка
и критерий разбиения подбираются по сетке во внутренней CV.

Все на train. При 71 рецидиве совместные эффекты и правила - гипотезы для
врачей, а не доказанные эффекты
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from . import config, models
from .eda import INK, short_name
from .importance import color_scale, original_feature, resample_index

# Исходный вариант дерева-правил, заданный вручную: для сравнения с подбором
RULE_DEPTH = 3  # не больше трех условий в правиле
RULE_LEAF = 10  # минимум пациентов в листе
# Сетка подбора: глубже 5 правило уже не прочитать, лист меньше 5 пациентов
# дает долю рецидива на 2-3 людях
RULE_GRID = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_leaf": [5, 10, 15, 20, 30],
    "ccp_alpha": [0, 0.005, 0.01, 0.02],  # обрезка веток с малым выигрышем
    "criterion": ["gini", "entropy"],
}
# Brier, а не log loss: в листе бывает доля рецидива ровно 0 или 1, и один
# пациент с другим исходом делает log loss бесконечным
RULE_SCORING = "neg_brier_score"
N_RESAMPLES = 200  # бутстрэп-ресэмплов для устойчивости правил


# ---------------------------------------------------------------- взаимодействия SHAP


def level_map(columns: list, features: list) -> np.ndarray:
    """Матрица столбцы модели x исходные признаки: 1, если столбец - уровень признака"""
    mapping = np.zeros((len(columns), len(features)))
    for i, column in enumerate(columns):
        mapping[i, features.index(original_feature(column))] = 1
    return mapping


def interaction_values(name: str, fitted, X: pd.DataFrame) -> np.ndarray:
    """Взаимодействия SHAP по исходным признакам: пациенты x признаки x признаки

    Уровни номинального признака складываются. Диагональ - вклад признака
    самого по себе, вне диагонали - половина вклада пары (вклад пары делится
    поровну между [i, j] и [j, i])
    """
    model = fitted.named_steps["model"]
    if name == "CatBoost":
        pool = Pool(fitted[:-1].transform(X), cat_features=config.NOMINAL)
        values = model.get_feature_importance(pool, type="ShapInteractionValues")
        values, columns = values[:, :-1, :-1], list(model.feature_names_)
    else:
        encoded = fitted.named_steps["prep"].transform(X)
        columns = list(fitted.named_steps["prep"].get_feature_names_out())
        values = shap.TreeExplainer(model).shap_interaction_values(encoded)
        if isinstance(values, list):
            values = values[1]  # класс 1 (рецидив)
        elif values.ndim == 4:
            values = values[..., 1]
    mapping = level_map(columns, list(X.columns))
    return np.einsum("nij,ia,jb->nab", values, mapping, mapping)


def top_pairs(values: np.ndarray, features: list, top: int = 10) -> pd.DataFrame:
    """Пары признаков по среднему модулю совместного вклада, самые сильные сверху

    Совместный вклад пары у пациента - сумма [i, j] и [j, i]
    """
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pair = values[:, i, j] + values[:, j, i]
            rows.append(
                {
                    "признак 1": features[i],
                    "признак 2": features[j],
                    "средний модуль вклада пары": np.abs(pair).mean(),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        "средний модуль вклада пары", ascending=False
    )
    return table.head(top).reset_index(drop=True)


def plot_pair(
    shap_values: pd.DataFrame, X: pd.DataFrame, first: str, second: str, title: str
) -> plt.Figure:
    """Вклад SHAP первого признака против его значения, цвет - значение второго

    shap_values - SHAP по исходным признакам. Если цвета расходятся по
    вертикали при одном значении первого признака, эффект первого зависит от
    второго. Пропуск второго признака - серый
    """
    color = color_scale(X[second])
    known = ~np.isnan(color)
    x = pd.to_numeric(X[first], errors="coerce").to_numpy()
    y = shap_values[first].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(x[~known], y[~known], s=14, color="lightgray")
    points = ax.scatter(
        x[known], y[known], s=14, c=color[known], cmap="coolwarm", vmin=0, vmax=1
    )
    ax.axhline(0, color="gray", lw=1)
    ax.set(xlabel=short_name(first), ylabel=f"вклад: {short_name(first)}", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    bar = fig.colorbar(points, ax=ax, shrink=0.7)
    bar.set_label(
        f"{short_name(second)}: синий - минимум, красный - максимум", fontsize=7
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- правила


def rules_model(**params) -> DecisionTreeClassifier:
    """Дерево-правила на сырых признаках: пропуски дерево обрабатывает само

    Заполнение медианой здесь вводило бы в заблуждение: у признака с частыми
    пропусками порог около медианы отделял бы пропуск от известных значений.
    Дерево sklearn само выбирает, в какую ветку идет пропуск, и это пишется
    в правиле
    """
    return DecisionTreeClassifier(random_state=config.RANDOM_SEED, **params)


def rules_search() -> GridSearchCV:
    """Подбор дерева-правил по сетке RULE_GRID во внутренней CV по Brier"""
    return GridSearchCV(
        rules_model(), RULE_GRID, scoring=RULE_SCORING, cv=models.inner_folds()
    )


def number(value: float) -> str:
    """Порог правила в русской записи"""
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def rules_table(
    tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Листья дерева-правил: условия, пациентов train, рецидивов и их доля

    Если у пациентов, дошедших до разбиения, признак бывает пропущен, в
    условии ветки, куда идет пропуск, дописано "или пропуск". Сортировка по
    доле рецидива: сверху правила, где рецидив чаще всего
    """
    structure = tree.tree_
    names = list(X.columns)
    membership = tree.decision_path(X).toarray().astype(bool)
    leaves = tree.apply(X)
    paths = {0: []}
    for node in range(
        structure.node_count
    ):  # потомки в дереве sklearn идут после родителя
        left, right = structure.children_left[node], structure.children_right[node]
        if left == -1:
            continue
        feature = names[structure.feature[node]]
        threshold = number(structure.threshold[node])
        has_missing = X.loc[membership[:, node], feature].isna().any()
        to_left = bool(structure.missing_go_to_left[node])
        left_text = f"{short_name(feature)} ≤ {threshold}"
        right_text = f"{short_name(feature)} > {threshold}"
        if has_missing:
            if to_left:
                left_text += " или пропуск"
            else:
                right_text += " или пропуск"
        paths[left] = [*paths[node], left_text]
        paths[right] = [*paths[node], right_text]
    rows = []
    outcome = y.to_numpy()
    for leaf in np.unique(leaves):
        members = leaves == leaf
        rows.append(
            {
                "правило": " и ".join(paths[leaf]),
                "пациентов": int(members.sum()),
                "рецидивов": int(outcome[members].sum()),
                "доля рецидива": outcome[members].mean(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("доля рецидива", ascending=False)
        .reset_index(drop=True)
    )


def rules_features(tree: DecisionTreeClassifier, features: list) -> set:
    """Признаки, по которым дерево-правила делает разбиения"""
    return {features[f] for f in tree.tree_.feature if f >= 0}


def rules_one(X, y, params: dict, seed: int) -> set:
    """Признаки дерева-правил с параметрами params на одном бутстрэп-ресэмпле"""
    idx = resample_index(len(y), seed)
    tree = rules_model(**params).fit(X.iloc[idx], y.iloc[idx])
    return rules_features(tree, list(X.columns))


def rules_stability(X: pd.DataFrame, y: pd.Series, params: dict) -> pd.Series:
    """Доля бутстрэп-ресэмплов, в которых признак входит в дерево-правила"""
    runs = Parallel(n_jobs=-1)(
        delayed(rules_one)(X, y, params, config.RANDOM_SEED + b)
        for b in range(N_RESAMPLES)
    )
    share = pd.Series({f: np.mean([f in run for run in runs]) for f in X.columns})
    return share[share > 0].sort_values(ascending=False)
