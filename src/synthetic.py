"""Генеративная синтетика (приложение Б, в статью не идет)

Вопрос шага: станет ли модель лучше, если дописать в обучающую часть
синтетических пациентов. Генератор учится только на обучающей части фолда,
модель проверяется на реальных пациентах валидационной части.

Генераторы:
- своя гауссова копула (ниже), для каждого класса отдельно;
- гауссова копула и CTGAN из пакета SDV (лицензия BUSL-1.1, использование
  разрешено, кроме коммерческого сервиса синтетических данных). Они учатся на
  всей обучающей части вместе с исходом, пациенты нужного класса порождаются
  по условию на исход. CTGAN - нейросеть-генератор (генеративно-состязательная
  сеть), может уловить нелинейные связи, но на малых данных нестабилен.
  Настройки SDV по умолчанию: копула описывает каждый количественный признак
  бета-распределением, CTGAN учится 300 эпох пакетами по 500 строк, при 183
  пациентах фолда это один шаг обучения на эпоху

Своя гауссова копула для каждого класса отдельно:
1. пропуски заполняются так же, как в модели: медиана для числовых, мода для
   номинальных (синтетические пациенты получаются без пропусков);
2. каждый признак переводится в нормальную шкалу через ранги (эмпирическая
   функция распределения), по этим значениям считается матрица корреляций;
3. новые пациенты: точки из многомерного нормального с этой матрицей
   корреляций, каждая координата возвращается в шкалу признака через квантили
   реальных значений.

Так сохраняются распределение каждого признака и попарные связи между ними.
Количественные признаки интерполируются между реальными значениями, у бинарных,
порядковых и номинальных берутся только коды из заполненных данных. Медиана
бинарного или порядкового признака при четном числе известных значений может
быть 0,5, тогда и этот код попадает в синтетику. У номинальных копула видит
порядок кодов, которого по смыслу нет, это упрощение. В SDV бинарные,
порядковые, номинальные признаки и исход объявлены категориальными,
количественные - числовыми

Прогон SDV с CTGAN идет долго, поэтому результаты CV вариантов синтетики
сохраняются в data/interim/step11 и читаются с диска при тех же данных и
настройках
"""

import json
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from . import config, evaluate
from .eda import INK, short_name
from .tuning import describe, fingerprint, read_info, replace_directory

# добавка к диагонали матрицы корреляций: защищает выборку из многомерного
# нормального от крошечных отрицательных собственных чисел из-за округления
RIDGE = 1e-6
DISCRETE = config.BINARY + config.ORDINAL + config.NOMINAL
OUTCOME = "исход"  # имя столбца исхода в таблице для SDV
GENERATORS = ["копула", "SDV копула", "SDV CTGAN"]
CACHE_DIR = config.INTERIM_DIR / "step11"
# версия кода генераторов в ключе сохранения: увеличивать при правке генерации,
# чтобы старые результаты не читались с диска
CODE_VERSION = 2


def check_aligned(X: pd.DataFrame, y: pd.Series) -> None:
    """Пациенты X и y должны идти в одном порядке с одинаковыми индексами"""
    if not X.index.equals(y.index):
        raise ValueError("Индексы X и y не совпадают")


def fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Пропуски как в модели: медиана для числовых, мода для номинальных"""
    filled = X.copy()
    for column in filled.columns:
        if column in config.NOMINAL:
            filled[column] = filled[column].fillna(filled[column].mode().iloc[0])
        else:
            filled[column] = filled[column].fillna(filled[column].median())
    return filled


def copula_sample(X: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n синтетических пациентов по гауссовой копуле, обученной на X без пропусков"""
    values = X.to_numpy(dtype=float)
    ranks = scipy_stats.rankdata(values, axis=0) / (len(values) + 1)
    normal = scipy_stats.norm.ppf(ranks)
    correlation = np.corrcoef(normal, rowvar=False)
    # столбец-константа дает NaN в корреляциях, такой столбец ни с чем не связан
    correlation = np.nan_to_num(correlation)
    np.fill_diagonal(correlation, 1.0 + RIDGE)
    draws = rng.multivariate_normal(np.zeros(len(correlation)), correlation, size=n)
    probabilities = scipy_stats.norm.cdf(draws)
    synthetic = {}
    for j, column in enumerate(X.columns):
        # дискретным - только коды из данных, количественным - интерполяция
        method = "inverted_cdf" if column in DISCRETE else "linear"
        synthetic[column] = np.quantile(
            values[:, j], probabilities[:, j], method=method
        )
    return pd.DataFrame(synthetic)


def copula_generate(X: pd.DataFrame, y: pd.Series, counts: dict, seed: int) -> tuple:
    """Синтетические пациенты своей копулой: counts - {класс: сколько сгенерировать}"""
    check_aligned(X, y)
    rng = np.random.default_rng(seed)
    filled = fill_missing(X)
    parts, labels = [], []
    for label, n in counts.items():
        if n > 0:
            parts.append(copula_sample(filled[y == label], n, rng))
            labels.append(pd.Series(label, index=range(n)))
    return pd.concat(parts, ignore_index=True), pd.concat(labels, ignore_index=True)


def fix_seeds(seed: int) -> None:
    """Зерно для numpy, random и torch: от них зависит обучение CTGAN

    Выборку из обученного генератора SDV задает свое зерно, его ставит
    sdv_generate
    """
    import torch  # тяжелый импорт, нужен только генераторам SDV

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sdv_generate(
    X: pd.DataFrame, y: pd.Series, counts: dict, generator: str, seed: int
) -> tuple:
    """Синтетические пациенты генератором SDV с условием на исход"""
    from sdv.metadata import Metadata
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

    check_aligned(X, y)
    table = fill_missing(X).reset_index(drop=True)
    table[OUTCOME] = y.to_numpy()
    metadata = Metadata.detect_from_dataframe(data=table, table_name="train")
    for column in table.columns:
        categorical = column in DISCRETE or column == OUTCOME
        metadata.update_column(
            column_name=column,
            table_name="train",
            sdtype="categorical" if categorical else "numerical",
        )
    fix_seeds(seed)
    if generator == "SDV копула":
        synthesizer = GaussianCopulaSynthesizer(metadata)
    elif generator == "SDV CTGAN":
        synthesizer = CTGANSynthesizer(metadata, enable_gpu=False)
    else:
        raise ValueError(f"Неизвестный генератор SDV {generator!r}")
    with warnings.catch_warnings():
        # совет сохранить метаданные в json: метаданные строятся заново в коде
        warnings.filterwarnings("ignore", message=".*save_to_json.*")
        synthesizer.fit(table)
    # без этого SDV перед выборкой ставит свое постоянное зерно 73251, и
    # выборка не зависела бы от seed. Метод приватный, версия SDV закреплена
    synthesizer._set_random_state(seed)
    conditions = [
        Condition(num_rows=n, column_values={OUTCOME: label})
        for label, n in counts.items()
        if n > 0
    ]
    sampled = synthesizer.sample_from_conditions(conditions).reset_index(drop=True)
    # при нехватке попыток SDV возвращает меньше строк с одним предупреждением
    produced = sampled[OUTCOME].value_counts().to_dict()
    wanted = {label: n for label, n in counts.items() if n > 0}
    if produced != wanted:
        raise RuntimeError(f"SDV выдал {produced} вместо {wanted}")
    return sampled[list(X.columns)], sampled[OUTCOME].astype(int)


def make_synthetic(
    X: pd.DataFrame, y: pd.Series, counts: dict, generator: str, seed: int
) -> tuple:
    """Синтетические пациенты генератором из GENERATORS"""
    if generator == "копула":
        return copula_generate(X, y, counts, seed)
    return sdv_generate(X, y, counts, generator, seed)


def synthetic_counts(y: pd.Series, mode: str) -> dict:
    """Сколько синтетических пациентов каждого класса добавить

    "удвоение" - столько же, сколько реальных, в каждом классе: доля рецидивов
    не меняется. "баланс" - только рецидивы (класс 1), до равенства классов
    (как SMOTENC на шаге 7), поэтому единичных эпизодов должно быть больше
    """
    counts = y.value_counts()
    if mode == "удвоение":
        return {label: int(n) for label, n in counts.items()}
    if mode == "баланс":
        if counts[0] <= counts[1]:
            raise ValueError("Для баланса единичных эпизодов должно быть больше")
        return {1: int(counts[0] - counts[1])}
    raise ValueError(f"Неизвестный вариант синтетики {mode!r}")


class SyntheticAugment(ClassifierMixin, BaseEstimator):
    """Модель, обучаемая на реальных и синтетических пациентах

    model - несобранная модель (пайплайн с предобработкой), mode - вариант из
    synthetic_counts, generator - один из GENERATORS. Синтетика создается
    заново при каждом fit только из переданных данных, поэтому в CV она не
    видит валидационных пациентов. Прогноз - обычный, по реальным пациентам
    """

    def __init__(
        self,
        model,
        mode: str = "удвоение",
        generator: str = "копула",
        random_state: int = config.RANDOM_SEED,
    ):
        self.model = model
        self.mode = mode
        self.generator = generator
        self.random_state = random_state

    def fit(self, X, y):
        """Обучение модели на реальных пациентах плюс синтетических"""
        counts = synthetic_counts(y, self.mode)
        synthetic_X, synthetic_y = make_synthetic(
            X, y, counts, self.generator, self.random_state
        )
        combined_X = pd.concat(
            [X.reset_index(drop=True), synthetic_X], ignore_index=True
        )
        combined_y = pd.concat(
            [y.reset_index(drop=True), synthetic_y], ignore_index=True
        )
        self.model_ = clone(self.model).fit(combined_X, combined_y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        """Вероятности классов от модели, обученной с синтетикой"""
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Классы при пороге по умолчанию (0,5)"""
        return self.model_.predict(X)


# ---------------------------------------------------------------- сохранение


def settings(model: SyntheticAugment, X: pd.DataFrame, y: pd.Series) -> dict:
    """Что должно совпасть, чтобы взять сохраненный результат варианта

    Данные (отпечаток), генератор, вариант, зерно, фолды, параметры модели,
    версия кода генераторов и версии пакетов, от которых зависит генерация
    """
    import copulas
    import ctgan
    import rdt
    import sdv
    import torch

    return {
        "генератор": model.generator,
        "вариант": model.mode,
        "данные": fingerprint(X, y),
        "фолды": [evaluate.N_FOLDS, evaluate.N_REPEATS],
        "зерно": model.random_state,
        "модель": {
            key: describe(value)
            for key, value in sorted(model.model.get_params(deep=True).items())
        },
        "версия кода": CODE_VERSION,
        "пакеты": {
            package.__name__: package.__version__
            for package in [sdv, rdt, ctgan, copulas, torch]
        },
    }


def cross_validate_cached(
    name: str, model: SyntheticAugment, X: pd.DataFrame, y: pd.Series
) -> dict:
    """evaluate.cross_validate варианта синтетики с сохранением на диск

    Результат берется с диска, если совпадают настройки (settings).
    Сохраняются OOF-прогнозы и ROC-AUC по фолдам, этого хватает для всех таблиц
    и графиков рамки оценки. Новый прогон целиком заменяет старый
    """
    directory = CACHE_DIR / name
    current = settings(model, X, y)
    info = read_info(directory, current)
    if info is not None:
        oof = pd.read_csv(directory / "oof.csv", index_col=0)
        oof.columns = oof.columns.astype(int)
        n_folds = len(info["ROC-AUC по фолдам"])
        return {
            "модель": name,
            "ROC-AUC по фолдам": info["ROC-AUC по фолдам"],
            "параметры по фолдам": [None] * n_folds,
            "пробы по фолдам": [None] * n_folds,
            "oof": oof.reindex(y.index),
            "y": y,
        }
    result = evaluate.cross_validate(name, model, X, y)
    info = {
        "настройки": json.loads(json.dumps(current)),
        "ROC-AUC по фолдам": [float(v) for v in result["ROC-AUC по фолдам"]],
    }

    def write(draft):
        result["oof"].to_csv(draft / "oof.csv")
        (draft / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    replace_directory(directory, write)
    return result


# ---------------------------------------------------------------- качество синтетики


def quality(
    X: pd.DataFrame, y: pd.Series, generator: str, seed: int = config.RANDOM_SEED
) -> tuple:
    """Насколько синтетика генератора похожа на реальных пациентов train

    Генератор учится на всем train (заполненном), синтетики столько же, сколько
    реальных в каждом классе. Возвращает таблицу KS по количественным признакам
    (статистика KS между реальными заполненными и синтетическими значениями) и
    пары корреляций Спирмена реальные - синтетические по всем парам признаков.
    Сравнение с теми же пациентами, на которых генератор учился, поэтому
    сходство оптимистично. Реальные значения заполнены медианой, поэтому KS
    сравнивает и пики на медиане у признаков с частыми пропусками
    """
    synthetic_X, _ = make_synthetic(
        X, y, synthetic_counts(y, "удвоение"), generator, seed
    )
    real = fill_missing(X).reset_index(drop=True)
    ks = pd.DataFrame(
        {
            "генератор": generator,
            "признак": [short_name(c) for c in config.QUANTITATIVE],
            "KS": [
                scipy_stats.ks_2samp(real[c], synthetic_X[c]).statistic
                for c in config.QUANTITATIVE
            ],
        }
    )
    real_corr = real.corr(method="spearman").to_numpy()
    synthetic_corr = synthetic_X[real.columns].corr(method="spearman").to_numpy()
    upper = np.triu_indices_from(real_corr, k=1)
    pairs = pd.DataFrame(
        {"реальные": real_corr[upper], "синтетические": synthetic_corr[upper]}
    )
    return ks, pairs


def plot_correlation_pairs(pairs: dict) -> plt.Figure:
    """Корреляции Спирмена пар признаков, реальные против синтетических, по генераторам

    pairs - {генератор: таблица пар из quality}. В заголовке панели -
    корреляция между реальными и синтетическими корреляциями. Пары без
    корреляции (столбец-константа) не рисуются, их число пишется в заголовке
    """
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4.6 * len(pairs), 4.6), sharey=True
    )
    for ax, (generator, table) in zip(np.atleast_1d(axes), pairs.items()):
        complete = table.dropna()
        agreement = np.corrcoef(complete["реальные"], complete["синтетические"])[0, 1]
        title = f"{generator}: r = {agreement:.2f}".replace(".", ",")
        if len(complete) < len(table):
            title += f"\nпар без корреляции: {len(table) - len(complete)}"
        ax.scatter(
            complete["реальные"],
            complete["синтетические"],
            s=12,
            color=evaluate.MODEL_COLORS[0],
            alpha=0.6,
        )
        ax.plot([-1, 1], [-1, 1], color="gray", lw=1, ls="--")
        ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            xlabel="корреляция у реальных пациентов",
            title=title,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)
    np.atleast_1d(axes)[0].set_ylabel("корреляция у синтетических пациентов")
    fig.tight_layout()
    return fig
