"""Подбор гиперпараметров деревьев Optuna во вложенной CV (шаг 8)

Внутри каждого внешнего фолда рамки оценки OptunaSearch подбирает параметры
по ROC-AUC внутренней стратифицированной CV на 5 фолдов, затем обучает модель с
лучшими параметрами на всей обучающей части фолда. Внешний фолд в подборе не
участвует, поэтому оценка вложенная.

Пространства заданы заново на нынешнем train. Пробы прежней версии проекта
(seed 42) не используем: тот train содержал большую часть нынешних тестовых
пациентов. После прогона проверяем, сошелся ли подбор и не упирается ли выбор в
края диапазонов. Диапазоны по итогам этой проверки не меняем: иначе подбор
подстраивался бы под ту же внешнюю CV, по которой оценивается модель.

Подбор трех моделей на 25 фолдах идет около часа, поэтому результаты
сохраняются в data/interim/step8 и при повторном прогоне читаются с диска.
Сохраненный результат берется, только если совпадают отпечаток данных (значения
признаков и исхода), настройки рамки оценки, пространство и число проб. Чтобы
пересчитать, папку достаточно удалить
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config, evaluate, models
from .eda import INK

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 40  # проб на внешний фолд
# Первые пробы случайные, дальше TPE строит модель пространства по ним. Зерно
# одно во всех внешних фолдах, поэтому случайные пробы в них одинаковы, а
# различаются фолды данными и тем, куда TPE идет дальше
N_STARTUP_TRIALS = 10
EDGE = 0.05  # выбор ближе 5% диапазона к краю считаем упором в край
CACHE_DIR = config.INTERIM_DIR / "step8"

# Пространства: ("int", от, до), ("float", от, до), ("log", от, до) - равномерно
# по логарифму, ("choice", [варианты])
SPACES = {
    "CatBoost": {
        "iterations": ("int", 100, 800),
        "learning_rate": ("log", 0.003, 0.3),
        # глубина 5 дает 32 листа, около 6 пациентов на лист, глубже нет смысла
        "depth": ("int", 1, 5),
        "l2_leaf_reg": ("log", 0.5, 30),
        "random_strength": ("log", 0.1, 10),
        # 2 - номинальные кодируются статистиками по исходу, 8 - через OHE
        "one_hot_max_size": ("choice", [2, 8]),
        # пропуск числового признака идет влево или вправо от всех значений
        "nan_mode": ("choice", ["Min", "Max"]),
    },
    "случайный лес": {
        "max_depth": ("int", 2, 12),
        "min_samples_leaf": ("int", 1, 30),
        "max_features": ("float", 0.1, 1.0),
        "max_samples": ("float", 0.4, 1.0),
    },
    "LightGBM": {
        "n_estimators": ("int", 50, 800),
        "learning_rate": ("log", 0.003, 0.3),
        # сложность дерева задает только глубина, число листьев закреплено в
        # models.lightgbm на 2 ** 6 и при глубине до 6 не ограничивает
        "max_depth": ("int", 1, 6),
        "min_child_samples": ("int", 3, 40),
        "subsample": ("float", 0.4, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "reg_lambda": ("log", 1e-4, 10),
    },
}
BUILDERS = {
    "CatBoost": models.catboost,
    "случайный лес": models.random_forest,
    "LightGBM": models.lightgbm,
}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """Параметры пробы Optuna по описанию пространства"""
    params = {}
    for name, (kind, *bounds) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, *bounds)
        elif kind == "float":
            params[name] = trial.suggest_float(name, *bounds)
        elif kind == "log":
            params[name] = trial.suggest_float(name, *bounds, log=True)
        elif kind == "choice":
            params[name] = trial.suggest_categorical(name, bounds[0])
        else:
            raise ValueError(f"Неизвестный тип параметра {kind!r} у {name}")
    return params


class OptunaSearch(ClassifierMixin, BaseEstimator):
    """Подбор параметров Optuna во внутренней CV и обучение лучшей модели

    build - функция, собирающая модель по параметрам (models.catboost и др.),
    space - пространство из SPACES. После fit доступны best_params_,
    best_score_ (ROC-AUC внутренней CV, оптимистичен), trials_ (все пробы) и
    best_estimator_. Внутренние фолды считаются параллельно, по процессу на фолд
    (n_jobs), сами модели обучаются в один поток
    """

    def __init__(
        self,
        build,
        space: dict,
        n_trials: int = N_TRIALS,
        inner_folds: int = models.INNER_FOLDS,
        n_jobs: int = models.INNER_FOLDS,
        random_state: int = config.RANDOM_SEED,
    ):
        self.build = build
        self.space = space
        self.n_trials = n_trials
        self.inner_folds = inner_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """Подбор во внутренней CV, затем обучение лучшей модели на всех X, y

        Ошибка обучения в пробе останавливает подбор (error_score="raise"), а не
        превращается молча в NaN
        """
        inner = StratifiedKFold(
            self.inner_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial):
            model = self.build(**suggest(trial, self.space))
            scores = cross_val_score(
                model,
                X,
                y,
                cv=inner,
                scoring="roc_auc",
                n_jobs=self.n_jobs,
                error_score="raise",
            )
            return scores.mean()

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=N_STARTUP_TRIALS, seed=self.random_state
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.trials_ = study.trials_dataframe(attrs=("number", "value", "params"))
        self.best_estimator_ = self.build(**self.best_params_).fit(X, y)
        self.classes_ = self.best_estimator_.classes_
        return self

    def predict_proba(self, X):
        """Вероятности классов от лучшей модели"""
        return self.best_estimator_.predict_proba(X)

    def predict(self, X):
        """Классы от лучшей модели при ее пороге по умолчанию (0,5)"""
        return self.best_estimator_.predict(X)


def space_table() -> pd.DataFrame:
    """Пространства подбора всех моделей одной таблицей для ноутбука"""
    kinds = {
        "int": "целое",
        "float": "равномерно",
        "log": "по логарифму",
        "choice": "вариант",
    }
    rows = []
    for model, space in SPACES.items():
        for name, (kind, *bounds) in space.items():
            if kind == "choice":
                values = ", ".join(str(v) for v in bounds[0])
            else:
                values = f"{bounds[0]:g} - {bounds[1]:g}"
            rows.append(
                {
                    "модель": model,
                    "параметр": name,
                    "как": kinds[kind],
                    "значения": values,
                }
            )
    return pd.DataFrame(rows)


def search(name: str) -> OptunaSearch:
    """Подбор для модели из SPACES по ее названию"""
    return OptunaSearch(BUILDERS[name], SPACES[name])


# ---------------------------------------------------------------- сохранение


def fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Отпечаток данных: меняется при правке любого значения, пациента или столбца"""
    values = pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=True)
    columns = pd.util.hash_pandas_object(pd.Index([*X.columns, y.name]))
    return f"{values.sum() % 2**64:x}-{columns.sum() % 2**64:x}"


def describe(value):
    """Стабильное текстовое описание параметра модели для сравнения настроек

    repr модели не годится: у функций в нем адрес в памяти, он меняется от
    запуска к запуску. Эстиматор описываем именем класса (его параметры и так
    идут отдельными ключами get_params), функцию - именем
    """
    if hasattr(value, "get_params"):
        return type(value).__name__
    if callable(value):
        return value.__name__
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    return repr(value)


def settings(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Что должно совпасть, чтобы взять сохраненный результат

    Кроме модели, пространства и числа проб это данные и все, от чего зависит
    результат подбора: фолды, зерно, фиксированные параметры моделей и импутация
    """
    return {
        "модель": name,
        "пространство": SPACES[name],
        "проб": N_TRIALS,
        "случайных проб": N_STARTUP_TRIALS,
        "данные": fingerprint(X, y),
        "фолды": [evaluate.N_FOLDS, evaluate.N_REPEATS, models.INNER_FOLDS],
        "зерно": config.RANDOM_SEED,
        "модель без подбора": {
            key: describe(value)
            for key, value in sorted(BUILDERS[name]().get_params(deep=True).items())
        },
    }


def read_info(directory, current: dict) -> dict | None:
    """Сохраненный info.json прогона, если его настройки совпадают с current

    current сравнивается в виде после записи в json: кортеж там становится
    списком, и без этого настройки никогда бы не совпали
    """
    info_path = directory / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["настройки"] != json.loads(json.dumps(current)):
        return None
    return info


def replace_directory(directory, write) -> None:
    """Пишет новый прогон во временную папку и только целиком заменяет старый

    write(draft) записывает файлы прогона в папку draft. Прерванный прогон не
    смешивает файлы разных прогонов: старая папка остается нетронутой, пока
    новая не записана полностью
    """
    draft = directory.with_name(f"{directory.name}.tmp")
    shutil.rmtree(draft, ignore_errors=True)
    draft.mkdir(parents=True)
    write(draft)
    shutil.rmtree(directory, ignore_errors=True)
    draft.rename(directory)


def cross_validate_cached(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """evaluate.cross_validate для подбора модели name с сохранением на диск

    Сохраняем OOF-прогнозы, ROC-AUC и параметры по фолдам и все пробы. Если
    настройки (settings) совпадают, результат читается с диска. Новый прогон
    пишется во временную папку и только целиком заменяет старый, поэтому
    прерванный прогон не смешивает файлы разных прогонов
    """
    directory = CACHE_DIR / name
    current = settings(name, X, y)
    info = read_info(directory, current)
    if info is not None:
        oof = pd.read_csv(directory / "oof.csv", index_col=0)
        oof.columns = oof.columns.astype(int)
        trials = pd.read_csv(directory / "trials.csv")
        return {
            "модель": name,
            "ROC-AUC по фолдам": info["ROC-AUC по фолдам"],
            "параметры по фолдам": info["параметры по фолдам"],
            "пробы по фолдам": [
                t.drop(columns="фолд") for _, t in trials.groupby("фолд")
            ],
            "oof": oof,
            "y": y,
        }
    result = evaluate.cross_validate(name, search(name), X, y)
    trials = pd.concat(
        [t.assign(фолд=i) for i, t in enumerate(result["пробы по фолдам"])]
    )
    info = {
        "настройки": json.loads(json.dumps(current)),
        "ROC-AUC по фолдам": [float(v) for v in result["ROC-AUC по фолдам"]],
        "параметры по фолдам": result["параметры по фолдам"],
    }

    def write(draft):
        result["oof"].to_csv(draft / "oof.csv")
        trials.to_csv(draft / "trials.csv", index=False)
        (draft / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    replace_directory(directory, write)
    return result


# ---------------------------------------------------------------- аудит подбора


def position(value, kind: str, low: float, high: float):
    """Положение значения (числа или столбца) в диапазоне: 0 - низ, 1 - верх"""
    if kind == "log":
        value, low, high = np.log(value), np.log(low), np.log(high)
    return (value - low) / (high - low)


def numeric_space(name: str) -> dict:
    """Числовые параметры пространства модели name, без вариантов-списков"""
    return {p: spec for p, spec in SPACES[name].items() if spec[0] != "choice"}


def boundary_table(result: dict) -> pd.DataFrame:
    """Где в диапазоне оказались выбранные значения числовых параметров

    Для каждого параметра: диапазон, медиана выбранных по внешним фолдам и доля
    фолдов, где выбор ближе EDGE к нижнему или верхнему краю
    """
    chosen = pd.DataFrame(result["параметры по фолдам"])
    rows = []
    for name, (kind, low, high) in numeric_space(result["модель"]).items():
        where = position(chosen[name], kind, low, high)
        rows.append(
            {
                "модель": result["модель"],
                "параметр": name,
                "от": low,
                "до": high,
                "медиана выбранных": chosen[name].median(),
                "у нижнего края": (where < EDGE).mean(),
                "у верхнего края": (where > 1 - EDGE).mean(),
            }
        )
    return pd.DataFrame(rows)


def choice_counts(result: dict) -> pd.DataFrame:
    """Сколько раз по внешним фолдам выбран каждый вариант параметров-списков"""
    chosen = pd.DataFrame(result["параметры по фолдам"])
    rows = []
    for name, (kind, *_) in SPACES[result["модель"]].items():
        if kind != "choice":
            continue
        for value, count in chosen[name].astype(str).value_counts().items():
            rows.append(
                {
                    "модель": result["модель"],
                    "параметр": name,
                    "вариант": value,
                    "фолдов": count,
                }
            )
    return pd.DataFrame(rows)


def convergence_table(results: list, checkpoint: int = 30) -> pd.DataFrame:
    """Растет ли лучший результат после пробы checkpoint

    По каждой модели: медиана по внешним фолдам лучшего ROC-AUC внутренней CV
    к пробе checkpoint и к последней пробе, и в скольких фолдах лучший
    результат после checkpoint еще улучшился
    """
    rows = []
    for result in results:
        best = [
            np.maximum.accumulate(t["value"].to_numpy())
            for t in result["пробы по фолдам"]
        ]
        rows.append(
            {
                "модель": result["модель"],
                f"медиана лучшего к пробе {checkpoint}": np.median(
                    [b[checkpoint - 1] for b in best]
                ),
                "медиана лучшего к последней пробе": np.median([b[-1] for b in best]),
                f"фолдов с ростом после пробы {checkpoint}": sum(
                    b[-1] > b[checkpoint - 1] for b in best
                ),
                "фолдов": len(best),
            }
        )
    return pd.DataFrame(rows)


def plot_convergence(results: list, colors: list) -> plt.Figure:
    """Лучший ROC-AUC внутренней CV по ходу проб, по панели на модель

    Тонкие линии - внешние фолды, толстая - медиана по фолдам. Если к последним
    пробам линия еще растет, проб мало. colors - цвета моделей, как на
    сравнительных графиках
    """
    fig, axes = plt.subplots(
        1, len(results), figsize=(5 * len(results), 4), sharey=True
    )
    for ax, color, result in zip(np.atleast_1d(axes), colors, results):
        curves = np.array(
            [
                np.maximum.accumulate(t["value"].to_numpy())
                for t in result["пробы по фолдам"]
            ]
        )
        trials = np.arange(1, curves.shape[1] + 1)
        for curve in curves:
            ax.plot(trials, curve, color=color, lw=0.6, alpha=0.35)
        ax.plot(
            trials,
            np.median(curves, axis=0),
            color=INK,
            lw=2,
            label="медиана по фолдам",
        )
        ax.set(title=result["модель"], xlabel="номер пробы")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("лучший ROC-AUC внутренней CV")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_chosen(results: list, colors: list) -> plt.Figure:
    """Положение выбранных значений в диапазонах числовых параметров

    Точка - выбор в одном внешнем фолде, 0 - нижний край диапазона, 1 -
    верхний. Серые полосы - зоны упора в край (EDGE). colors - цвета моделей
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    rng = np.random.default_rng(config.RANDOM_SEED)  # разброс точек по вертикали
    for ax, color, result in zip(np.atleast_1d(axes), colors, results):
        space = numeric_space(result["модель"])
        chosen = pd.DataFrame(result["параметры по фолдам"])
        for row, (name, (kind, low, high)) in enumerate(space.items()):
            where = position(chosen[name], kind, low, high)
            jitter = rng.uniform(-0.15, 0.15, len(where))
            ax.scatter(where, row + jitter, s=14, color=color, alpha=0.7)
        ax.axvspan(0, EDGE, color="lightgray", zorder=0)
        ax.axvspan(1 - EDGE, 1, color="lightgray", zorder=0)
        ax.set_yticks(range(len(space)), list(space), fontsize=8)
        ax.set(
            title=result["модель"], xlabel="положение в диапазоне", xlim=(-0.02, 1.02)
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
