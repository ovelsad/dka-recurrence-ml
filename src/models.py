"""Фабрики моделей (шаги 3-8 и приложение)

Вся предобработка живет внутри пайплайна модели, поэтому в кросс-валидации она
заново обучается на каждом фолде и не видит валидационных пациентов.
Импорт enable_iterative_imputer включает экспериментальный IterativeImputer (MICE)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

KNN_NEIGHBORS = 5
MICE_ITERATIONS = 10  # число кругов уточнения в MICE, значение sklearn по умолчанию
IMPUTATION_METHODS = ["KNN", "MICE", "медиана"]  # кандидаты шага 4
IMPUTATION = "медиана"  # выбрана на шаге 4
# Проверки шага 4: откуда выигрыш медианы. В выбор метода не входят
IMPUTATION_CHECKS = [
    "KNN с индикаторами пропусков",
    "KNN, бинарные и порядковые медианой",
]
# Сила регуляризации, от 0,001 до 10, чем меньше C, тем сильнее штраф
C_GRID = np.logspace(-3, 1, 9)
L1_RATIOS = [0.25, 0.5, 0.75]  # доля L1 в elasticnet
INNER_FOLDS = 5
MAX_ITER = 5000  # с запасом, чтобы решатель логрега сходился на всех фолдах
# Число деревьев леса не подбираем: больше деревьев не переобучает, только
# стабилизирует прогноз, 500 хватает с запасом для 200 пациентов
FOREST_TREES = 500
# Дерево глубины d имеет не больше 2 ** d листьев. При глубине до 6 число
# листьев 64 ничего не ограничивает, и сложность LightGBM задает одна глубина
LIGHTGBM_LEAVES = 2**6

# Количественные, бинарные и порядковые идут одним числом, номинальные - OHE
NUMERIC = config.QUANTITATIVE + config.BINARY + config.ORDINAL


def imputer(method: str):
    """Импутер числовых признаков по названию метода

    KNN - среднее по 5 ближайшим пациентам, MICE - последовательная регрессия
    каждого признака на остальные, медиана - медиана train
    """
    if method == "KNN":
        return KNNImputer(n_neighbors=KNN_NEIGHBORS)
    if method == "MICE":
        return IterativeImputer(
            max_iter=MICE_ITERATIONS, random_state=config.RANDOM_SEED
        )
    if method == "медиана":
        return SimpleImputer(strategy="median")
    raise ValueError(f"Неизвестный метод импутации {method!r}")


def numeric_block(method: str):
    """Масштаб и импутация числовых признаков

    Масштабируем до импутации, чтобы расстояния между пациентами не зависели от
    единиц измерения. Бинарные и порядковые KNN и MICE заполняют дробным
    значением, для логрега это допустимо. Медиана обычно дает существующее
    значение, но при четном числе наблюдений может попасть между уровнями: на
    train так вышло с ХБП А и ХБП С в 3 из 25 фолдов (значение 0,5).
    Два варианта из IMPUTATION_CHECKS нужны только для проверок шага 4
    """
    if method == "KNN с индикаторами пропусков":
        # к заполненным значениям добавляются столбцы 0/1 "значение было пропущено"
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("impute", KNNImputer(n_neighbors=KNN_NEIGHBORS, add_indicator=True)),
            ]
        )
    if method == "KNN, бинарные и порядковые медианой":
        # соседей ищем только по количественным, остальные заполняем медианой
        return ColumnTransformer(
            [
                ("quantitative", numeric_block("KNN"), config.QUANTITATIVE),
                (
                    "categorical",
                    numeric_block("медиана"),
                    config.BINARY + config.ORDINAL,
                ),
            ]
        )
    return Pipeline([("scale", StandardScaler()), ("impute", imputer(method))])


def preprocessing(method: str, features: list | None = None) -> ColumnTransformer:
    """Числовые признаки: масштаб и импутация, номинальные: мода и OHE

    features - какие признаки модели брать (по умолчанию все), остальные
    столбцы X отбрасываются. Номинальные (тип СД, вид инсулинотерапии) в поиске
    соседей не участвуют и заполняются модой без индикатора пропуска. OHE без
    drop дает коллинеарные столбцы, при штрафе это работает, но коэффициенты
    этих столбцов по отдельности как ОШ не читаются
    """
    features = config.FEATURES if features is None else features
    # все, кроме номинальных, идет числом, включая служебные столбцы вроде
    # признака-зонда шага 8
    numeric = [f for f in features if f not in config.NOMINAL]
    nominal = [f for f in features if f in config.NOMINAL]
    nominal_block = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    blocks = []
    if numeric:
        blocks.append(("numeric", numeric_block(method), numeric))
    if nominal:
        blocks.append(("nominal", nominal_block, nominal))
    return ColumnTransformer(blocks)


def classifier(**params) -> LogisticRegression:
    """Логистическая регрессия проекта, params передаются в LogisticRegression

    random_state нужен решателю saga (L1 и elasticnet): он перемешивает строки,
    без зерна результаты подбора могут чуть различаться между запусками
    """
    return LogisticRegression(
        max_iter=MAX_ITER, random_state=config.RANDOM_SEED, **params
    )


def baseline_logreg(method: str = IMPUTATION) -> Pipeline:
    """Логистическая регрессия с параметрами по умолчанию (L2-штраф, C = 1)

    По умолчанию медиана для числовых и самая частая категория для номинальных:
    так устроен бейзлайн шага 3. На шаге 4 метод импутации меняется для сравнения
    """
    return Pipeline(
        [
            ("prep", preprocessing(method)),
            ("model", classifier()),
        ]
    )


def logreg(features: list | None = None, **params) -> Pipeline:
    """Логрег на признаках features (по умолчанию все) с импутацией медианой

    params передаются в LogisticRegression
    """
    return Pipeline(
        [
            ("prep", preprocessing(IMPUTATION, features)),
            ("model", classifier(**params)),
        ]
    )


def logreg_smote() -> ImbPipeline:
    """Логрег с SMOTENC: синтетические рецидивы добавляются только в обучающую часть

    SMOTENC - единственный вариант SMOTE в imblearn для смешанных данных:
    количественные интерполирует между соседями, а категориальным дает самое
    частое значение среди соседей, поэтому новых кодов, которых нет в обучающей
    части, не появляется. Бинарные и порядковые объявлены категориальными, иначе
    появилась бы, например, ретинопатия 1,4. Номинальные идут одним столбцом с
    кодом, OHE после SMOTENC, иначе уровни одного признака интерполировались бы
    независимо. Импутация та же, что в остальных вариантах, значение 0,5 от
    медианы (см. numeric_block) SMOTENC считает еще одним уровнем. Масштаб до
    SMOTENC, чтобы соседи не определялись единицами измерения. На тесте и
    валидации сэмплер не работает, только при обучении
    """
    prep = ColumnTransformer(
        [
            ("numeric", numeric_block(IMPUTATION), NUMERIC),
            ("nominal", SimpleImputer(strategy="most_frequent"), config.NOMINAL),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")  # имена столбцов нужны SMOTENC
    onehot = ColumnTransformer(
        [("nominal", OneHotEncoder(handle_unknown="ignore"), config.NOMINAL)],
        remainder="passthrough",
    )
    return ImbPipeline(
        [
            ("prep", prep),
            (
                "smote",
                SMOTENC(
                    categorical_features=config.BINARY
                    + config.ORDINAL
                    + config.NOMINAL,
                    random_state=config.RANDOM_SEED,
                ),
            ),
            ("onehot", onehot),
            ("model", classifier()),
        ]
    )


def logreg_grid() -> dict:
    """Сетка шага 6: сила штрафа C при L2

    L2 не обнуляет коэффициенты, поэтому модель А остается на всех признаках.
    L1 и elastic net обнуляют, они идут методами отбора шага 7
    """
    return {"model__C": C_GRID, "model__penalty": ["l2"], "model__solver": ["lbfgs"]}


def inner_folds() -> StratifiedKFold:
    """Внутренняя CV для подбора и отбора внутри внешнего фолда"""
    return StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


# Критерий подбора C. ROC-AUC зависит только от порядка пациентов, а сила
# штрафа этот порядок почти не меняет, поэтому по ROC-AUC C выбирается шумом.
# Log loss отвечает и за порядок, и за точность вероятностей
C_SCORING = "neg_log_loss"


def logreg_search(n_jobs: int = -1) -> GridSearchCV:
    """Подбор C логрега на всех признаках во внутренней CV по log loss

    Внутри внешних фолдов рамки оценки это дает вложенную CV. После подбора
    модель переобучается на всей переданной выборке с лучшими параметрами
    """
    return GridSearchCV(
        logreg(), logreg_grid(), scoring=C_SCORING, cv=inner_folds(), n_jobs=n_jobs
    )


# ---------------------------------------------------------------- деревья (шаг 8)
# Деревья обучаются в один поток: параллельно идут фолды внутренней CV


def nominal_as_text(X: pd.DataFrame) -> pd.DataFrame:
    """Номинальные признаки строкой для CatBoost, пропуск - отдельный уровень

    CatBoost не принимает NaN в категориальных признаках, поэтому пропуск типа
    СД становится уровнем "пропуск" и модель видит его напрямую (решение автора
    2026-09-25). Числовые признаки остаются с NaN, их CatBoost обрабатывает сам
    """
    X = X.copy()
    for column in [c for c in config.NOMINAL if c in X.columns]:
        codes = X[column].dropna()
        if not (codes == codes.round()).all():
            raise ValueError(f"В номинальном признаке {column} есть дробный код")
        X[column] = X[column].map(lambda v: "пропуск" if pd.isna(v) else str(int(v)))
    return X


def catboost(features: list | None = None, **params) -> Pipeline:
    """CatBoost с нативными пропусками, номинальные идут категориальными

    features - признаки X, на которых учится модель (по умолчанию все). CatBoost
    берет все столбцы переданного X, поэтому X передается уже с этими признаками
    """
    features = config.FEATURES if features is None else features
    return Pipeline(
        [
            ("text", FunctionTransformer(nominal_as_text)),
            (
                "model",
                CatBoostClassifier(
                    cat_features=[c for c in config.NOMINAL if c in features],
                    random_seed=config.RANDOM_SEED,
                    thread_count=1,
                    verbose=0,
                    allow_writing_files=False,
                    **params,
                ),
            ),
        ]
    )


def random_forest(features: list | None = None, **params) -> Pipeline:
    """Случайный лес на импутации шага 4 (медиана и мода, OHE номинальных)

    features - признаки, на которых учится модель (по умолчанию все). Масштаб из
    общей предобработки лесу не нужен, но и не мешает: деревья зависят только
    от порядка значений
    """
    return Pipeline(
        [
            ("prep", preprocessing(IMPUTATION, features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=FOREST_TREES,
                    random_state=config.RANDOM_SEED,
                    n_jobs=1,
                    **params,
                ),
            ),
        ]
    )


def lightgbm(features: list | None = None, **params) -> Pipeline:
    """LightGBM на импутации шага 4, как у леса

    features - признаки, на которых учится модель (по умолчанию все).
    subsample_freq = 1 включает подвыборку строк на каждом дереве, иначе
    параметр subsample не действует. Число листьев закреплено на LIGHTGBM_LEAVES,
    сложность дерева задает глубина
    """
    return Pipeline(
        [
            ("prep", preprocessing(IMPUTATION, features)),
            (
                "model",
                LGBMClassifier(
                    num_leaves=LIGHTGBM_LEAVES,
                    subsample_freq=1,
                    random_state=config.RANDOM_SEED,
                    n_jobs=1,
                    verbose=-1,
                    **params,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------- TabPFN (приложение А)


def tabpfn() -> Pipeline:
    """TabPFN v2: заранее обученная нейросеть для таблиц, считает локально

    Гиперпараметры не подбираем: модель не учится на наших данных в обычном
    смысле, а получает обучающую выборку как контекст и сразу дает прогноз.
    Пропуски обрабатывает сама, как CatBoost. Номинальные признаки ставим в
    конец и объявляем категориальными. Остальные TabPFN проверяет сам: при
    обучающей выборке больше 100 пациентов столбец с менее чем 4 разными
    значениями (пропуск тоже считается значением) тоже идет категориальным.
    Так категориальными становятся бинарные и часть порядковых, и у одного
    признака тип может меняться от фолда к фолду. Автоопределение оставлено
    как в модели по умолчанию. Веса v2
    скачиваются с HuggingFace без аккаунта, данные пациентов никуда не уходят.
    Импорт внутри функции: torch тяжелый и нужен только приложению А
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    order = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC),
            ("nominal", "passthrough", config.NOMINAL),
        ]
    )
    nominal = list(range(len(NUMERIC), len(NUMERIC) + len(config.NOMINAL)))
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        categorical_features_indices=nominal,
        device="cpu",
        random_state=config.RANDOM_SEED,
    )
    return Pipeline([("order", order), ("model", model)])
