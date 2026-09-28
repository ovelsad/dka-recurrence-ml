"""Важность признаков для статьи (шаг 10)

Все считаем на train, тест закрыт. Основной расчет - на моделях, обученных на
всем train: два логрега (все признаки и набор голосования, шаги 6-7) и деревья с
параметрами, которые Optuna подобрала на всем train.

Меры важности у каждой модели свои (ОШ логрега здесь не считаем: при малом числе
рецидивов и отборе столбцов внутри модели они неустойчивы, размеры эффекта дает
статистика шага 1):
- LightGBM: gain (суммарный выигрыш в разбиениях по признаку) и split (число
  разбиений по признаку)
- CatBoost: PredictionValuesChange (насколько в среднем меняется прогноз при
  изменении признака)
- случайный лес: снижение неоднородности Джини (MDI)
- SHAP у всех четырех: средний модуль вклада признака в прогноз пациента

Основной рейтинг строим по SHAP, по одной мере на модель, встроенные меры
справочно: у LightGBM их две, и медиана по всем мерам сместилась бы к нему,
а gain, split и MDI сами завышают признаки с множеством разных значений.
Единицы SHAP у моделей разные (у логрега, LightGBM и CatBoost - логит, у леса -
вероятность), поэтому модели сравниваем по рангам. Уровни номинального признака
после OHE суммируем в один исходный признак: у SHAP вклады аддитивны, у gain,
split и MDI сумма по уровням дает вклад признака целиком.

Устойчивость считаем двумя способами:
- бутстрэп train при фиксированных параметрах: модель обучается заново на
  ресэмпле, SHAP считается на исходном train. Изменчивость подбора параметров
  сюда не входит
- по 25 внешним фолдам шага 8 (для сравнения): в каждом фолде модель с
  параметрами, выбранными в этом фолде, обучается на его обучающей части, SHAP
  считается на валидационной, то есть на пациентах, которых модель не видела.
  Сюда входит и изменчивость подбора
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool
from joblib import Parallel, delayed
from sklearn.base import clone

from . import config, models, tuning
from .eda import INK, annotate_cells, short_name
from .evaluate import outer_folds

# бутстрэп-ресэмплов с переобучением модели для устойчивости (не путать с
# evaluate.N_BOOTSTRAP, ресэмплами прогнозов для ДИ)
N_REFITS = 200
TOP_SIZE = 5  # размер топа для устойчивости
PREP_BLOCKS = ("numeric", "nominal")  # префиксы столбцов предобработки


def original_feature(column: str) -> str:
    """Исходный признак по имени столбца после предобработки

    numeric__HbA1c -> HbA1c, nominal__<признак>_1.0 -> <признак>. Имена без
    известного префикса (CatBoost видит исходные признаки) возвращаются как есть
    """
    block, separator, rest = column.partition("__")
    if not separator or block not in PREP_BLOCKS:
        return column
    return rest.rsplit("_", 1)[0] if block == "nominal" else rest


def to_features(grouped: pd.Series, features: list) -> pd.Series:
    """Значения по исходным признакам, признаки без вклада получают 0

    Имя, которого нет среди признаков, - ошибка разбора имен столбцов, а не
    нулевая важность, поэтому падаем, а не теряем его молча
    """
    unknown = set(grouped.index) - set(features)
    if unknown:
        raise ValueError(f"Столбцы не сопоставлены с признаками: {sorted(unknown)}")
    return grouped.reindex(features, fill_value=0.0)


def by_feature(values: pd.Series, features: list) -> pd.Series:
    """Сумма по уровням до исходных признаков"""
    grouped = values.groupby(values.index.map(original_feature)).sum()
    return to_features(grouped, features)


def resample_index(n: int, seed: int) -> np.ndarray:
    """Номера строк бутстрэп-ресэмпла: n строк с возвращением"""
    return np.random.default_rng(seed).integers(0, n, n)


# ---------------------------------------------------------------- логрег


def logreg_columns(fitted) -> tuple[list, np.ndarray]:
    """Отобранные столбцы логрега и их коэффициенты"""
    names = list(fitted[:-1].get_feature_names_out())
    return names, fitted.named_steps["model"].coef_[0]


def logreg_shap(fitted, X: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """SHAP логрега: коэффициент на отклонение столбца от среднего, в логитах

    Среднее берется по background - пациентам, на которых модель обучалась.
    Так точка отсчета та же, что у TreeSHAP деревьев (обучающие данные)
    """
    names, coef = logreg_columns(fitted)
    encoded = fitted[:-1].transform(X)
    reference = fitted[:-1].transform(background).mean(axis=0)
    return pd.DataFrame((encoded - reference) * coef, index=X.index, columns=names)


# ---------------------------------------------------------------- деревья


def native_importance(name: str, fitted, X: pd.DataFrame) -> dict:
    """Встроенные меры важности модели name по исходным признакам"""
    features = list(X.columns)
    model = fitted.named_steps["model"]
    if name == "CatBoost":
        # имена берем из модели: так порядок столбцов X не может их перепутать
        values = model.get_feature_importance(type="PredictionValuesChange")
        return {
            "PredictionValuesChange": by_feature(
                pd.Series(values, index=model.feature_names_), features
            )
        }
    columns = fitted.named_steps["prep"].get_feature_names_out()
    if name == "LightGBM":
        booster = model.booster_
        return {
            kind: by_feature(
                pd.Series(booster.feature_importance(kind), index=columns), features
            )
            for kind in ("gain", "split")
        }
    if name == "случайный лес":
        return {
            "MDI": by_feature(
                pd.Series(model.feature_importances_, index=columns), features
            )
        }
    raise ValueError(f"Нет встроенной важности для модели {name!r}")


def tree_shap(name: str, fitted, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP дерева: пациенты x столбцы модели

    CatBoost и LightGBM считают точный TreeSHAP сами, в логитах, лес - через
    shap.TreeExplainer, в вероятности класса 1
    """
    model = fitted.named_steps["model"]
    if name == "CatBoost":
        catboost_input = fitted[:-1].transform(X)
        pool = Pool(catboost_input, cat_features=config.NOMINAL)
        values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        return pd.DataFrame(values, index=X.index, columns=model.feature_names_)
    encoded = fitted.named_steps["prep"].transform(X)
    columns = fitted.named_steps["prep"].get_feature_names_out()
    if name == "LightGBM":
        values = model.predict(encoded, pred_contrib=True)[:, :-1]
    elif name == "случайный лес":
        values = shap.TreeExplainer(model).shap_values(encoded)
        values = values[..., 1] if values.ndim == 3 else values  # класс 1
    else:
        raise ValueError(f"Нет SHAP для модели {name!r}")
    return pd.DataFrame(values, index=X.index, columns=columns)


def sum_levels(values: pd.DataFrame) -> pd.DataFrame:
    """SHAP по исходным признакам: вклады уровней номинального складываются у пациента"""
    return values.T.groupby(values.columns.map(original_feature)).sum().T


def shap_by_feature(values: pd.DataFrame, features: list) -> pd.Series:
    """Средний модуль SHAP по исходным признакам

    Вклады уровней одного номинального признака сначала складываются у каждого
    пациента, потом берется модуль: так признак оценивается целиком
    """
    return to_features(sum_levels(values).abs().mean(), features)


def shap_values(
    name: str, fitted, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame:
    """SHAP модели name на X: пациенты x столбцы модели

    background - пациенты, на которых модель обучалась. Логрегу он задает
    точку отсчета, деревья берут ее из своих обучающих данных сами
    """
    if name.startswith("логрег"):
        return logreg_shap(fitted, X, background)
    return tree_shap(name, fitted, X)


def shap_by_missing(
    values: pd.DataFrame, X: pd.DataFrame, features: list
) -> pd.DataFrame:
    """Средний SHAP признака отдельно у пациентов с пропуском и с известным значением

    values - SHAP по исходным признакам (уровни номинальных сложены). Показывает,
    идет ли вклад признака от самого пропуска
    """
    rows = []
    for feature in features:
        missing = X[feature].isna()
        for label, mask in [("пропуск", missing), ("известно", ~missing)]:
            rows.append(
                {
                    "признак": short_name(feature),
                    "значение": label,
                    "пациентов": int(mask.sum()),
                    "средний SHAP": values.loc[mask, feature].mean(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- сводка


def rank_table(importances: dict) -> pd.DataFrame:
    """Ранги признаков по каждой мере (1 - самый важный) и медианный ранг

    importances - {мера: Series по исходным признакам}. Признак с нулевой
    важностью (в модели его нет) получает последний ранг, равный числу
    признаков, ничьи среди ненулевых - средний ранг
    """
    ranks = {}
    for measure, values in importances.items():
        rank = values.rank(ascending=False, method="average")
        ranks[measure] = rank.where(values > 0, float(len(values)))
    table = pd.DataFrame(ranks)
    table["медианный ранг"] = table.median(axis=1)
    return table.sort_values("медианный ранг")


def in_top(values: pd.Series) -> pd.Series:
    """Вошел ли признак в топ-TOP_SIZE, нулевая важность в топ не входит

    Ничьи решает порядок столбцов (method="first"), на практике они бывают
    только у нулевых важностей, а те в топ не входят
    """
    return (values.rank(ascending=False, method="first") <= TOP_SIZE) & (values > 0)


def bootstrap_one(name: str, build, X, y, seed: int) -> pd.Series:
    """Средний модуль SHAP на исходном X после обучения на одном ресэмпле"""
    idx = resample_index(len(y), seed)
    train = X.iloc[idx]
    fitted = clone(build).fit(train, y.iloc[idx])
    return shap_by_feature(shap_values(name, fitted, X, train), list(X.columns))


def top_frequency(name: str, build, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Доля бутстрэп-ресэмплов, где признак в топ-TOP_SIZE по среднему модулю SHAP"""
    runs = Parallel(n_jobs=-1)(
        delayed(bootstrap_one)(name, build, X, y, config.RANDOM_SEED + b)
        for b in range(N_REFITS)
    )
    return (
        pd.DataFrame([in_top(run) for run in runs]).mean().sort_values(ascending=False)
    )


def fold_model(name: str, params: dict):
    """Несобранная модель name с параметрами, выбранными во внешнем фолде

    У логрега это отобранные признаки и параметры сетки (шаги 6-7, ключ "признаки"
    и параметры LogisticRegression), у деревьев - параметры Optuna шага 8
    """
    if name.startswith("логрег"):
        params = dict(params)
        return models.logreg(params.pop("признаки"), **params)
    return tuning.BUILDERS[name](**params)


def fold_shap_one(name: str, params: dict, X, y, train_idx, valid_idx) -> pd.Series:
    """Средний модуль SHAP на валидационной части одного внешнего фолда"""
    train = X.iloc[train_idx]
    fitted = fold_model(name, params).fit(train, y.iloc[train_idx])
    valid = X.iloc[valid_idx]
    return shap_by_feature(shap_values(name, fitted, valid, train), list(X.columns))


def fold_shap(
    name: str, fold_params: list, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Средний модуль SHAP по 25 внешним фолдам: фолды x признаки

    fold_params - параметры, выбранные в каждом внешнем фолде (результат
    cross_validate шагов 7 и 8), фолды те же, что в рамке оценки
    """
    splits = list(outer_folds().split(X, y))
    if len(splits) != len(fold_params):
        raise ValueError("Число наборов параметров не совпадает с числом фолдов")
    runs = Parallel(n_jobs=-1)(
        delayed(fold_shap_one)(name, params, X, y, train_idx, valid_idx)
        for params, (train_idx, valid_idx) in zip(fold_params, splits)
    )
    return pd.DataFrame(runs)


def heatmap_text(value: float, digits: int) -> str:
    """Число в ячейке тепловой карты в русской записи, целое без дробной части"""
    text = f"{value:.0f}" if float(value).is_integer() else f"{value:.{digits}f}"
    return text.replace(".", ",")


def plot_ranks(ranks: pd.DataFrame, title: str, shown: int = 12) -> plt.Figure:
    """Тепловая карта рангов признаков по мерам, первые shown по медианному рангу"""
    table = ranks.head(shown)
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 3.5, 0.4 * len(table) + 1.8))
    image = ax.imshow(values, cmap="Blues_r", vmin=1, vmax=len(ranks), aspect="auto")
    texts = [[heatmap_text(value, 1) for value in row] for row in values]
    annotate_cells(ax, texts, values <= len(ranks) / 3)  # у малых рангов фон темный
    ax.set_xticks(
        range(table.shape[1]), table.columns, rotation=35, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(table)), [short_name(f) for f in table.index], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig


def plot_top_frequency(
    frequency: pd.DataFrame, title: str, shown: int = 12
) -> plt.Figure:
    """Тепловая карта: доля ресэмплов или фолдов, где признак в топе, по моделям"""
    table = frequency.head(shown)
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 3.5, 0.4 * len(table) + 1.6))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    texts = [[heatmap_text(value, 2) for value in row] for row in values]
    annotate_cells(ax, texts, values > 0.6)  # выше 0,6 фон уже темный
    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
    ax.set_yticks(range(len(table)), [short_name(f) for f in table.index], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig


def color_scale(values: pd.Series) -> np.ndarray:
    """Значения признака в шкалу цвета от 0 (минимум) до 1 (максимум)

    Пропуск остается NaN, у признака с одним значением все точки посередине (0,5)
    """
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    span = np.nanmax(raw) - np.nanmin(raw)
    if span == 0:
        return np.where(np.isnan(raw), np.nan, 0.5)
    return (raw - np.nanmin(raw)) / span


def plot_shap_beeswarm(
    values: pd.DataFrame, X: pd.DataFrame, title: str, shown: int = 10
) -> plt.Figure:
    """SHAP по пациентам для первых shown признаков: точка - пациент, цвет - значение

    values - SHAP по исходным признакам (уровни номинальных уже сложены).
    Признаки с нулевым вкладом не показываем. Цвет - значение признака от
    минимума (синий) до максимума (красный), так бинарные остаются двумя
    цветами. Пропуск - светло-серый, известный уровень номинального -
    темно-серый: порядка у номинальных нет
    """
    importance = values.abs().mean()
    order = importance[importance > 0].sort_values(ascending=False).index[:shown]
    rng = np.random.default_rng(config.RANDOM_SEED)  # разброс точек по вертикали
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.6))
    points = None
    for row, feature in enumerate(reversed(order)):
        contribution = values[feature].to_numpy()
        color_value = color_scale(X[feature])
        jitter = rng.uniform(-0.25, 0.25, len(contribution))
        known = ~np.isnan(color_value)
        ax.scatter(contribution[~known], row + jitter[~known], s=8, color="lightgray")
        if feature in config.NOMINAL:
            ax.scatter(
                contribution[known], row + jitter[known], s=8, color=INK, alpha=0.6
            )
            continue
        points = ax.scatter(
            contribution[known],
            row + jitter[known],
            s=8,
            c=color_value[known],
            cmap="coolwarm",
            vmin=0,
            vmax=1,
        )
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(
        range(len(order)), [short_name(f) for f in reversed(order)], fontsize=8
    )
    ax.set(title=title, xlabel="вклад в прогноз (SHAP)")
    ax.spines[["top", "right"]].set_visible(False)
    if points is not None:
        bar = fig.colorbar(points, ax=ax, shrink=0.6)
        bar.set_label(
            "значение признака: синий - минимум, красный - максимум;\n"
            "светло-серый - пропуск, темно-серый - уровень номинального",
            fontsize=7,
        )
    fig.tight_layout()
    return fig
