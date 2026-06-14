"""Базовая модель и роль предобработки (этап 15).

Отвечает на вопрос рецензента: а с чем мы вообще сравниваем, помогла ли подготовка
данных. Базовая модель - простая логистическая регрессия без регуляризации, без
балансировки, обученная на сырых данных с обязательным минимумом (парсинг текста в
числа, заполнение пропусков median/мода, one-hot, масштабирование для сходимости).
Бустинг в качестве базовой модели не берем: это уже сложный метод.

Уровни (все оценены одинаково: пулированный out-of-fold ROC-AUC на одних и тех же
218 обучающих пациентах, пятифолдовая стратифицированная CV):

1. Базовая логрег без регуляризации на сырых данных (аномалии не исправлены,
   колонки не отобраны, все признаки кроме идентификаторов, дат и колонки-утечки).
2. Та же базовая логрег на очищенных данных, набор без мультиколлинеарности.
   Разница с уровнем 1 - вклад очистки и отбора признаков.
3. Полный пайплайн (тюнингованные финалисты) - из готовых таблиц: регуляризация,
   KNN-импутация, взвешивание классов, подбор гиперпараметров.

Идентификаторы, даты (временной артефакт сбора) и колонка-утечка
"Количество ДКА в анамнезе" (по ней ставили целевую переменную) исключены везде.

Запуск: python -m src.baseline_ladder (в окружении dka).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import cleaning, config, features, io
from .config import RANDOM_SEED

# Колонки, которые нельзя подавать ни в один baseline:
# идентификаторы, даты (временной артефакт сбора, не клинический предиктор) и
# колонка-утечка, по которой ставили целевую переменную.
LEAKAGE = "Количество ДКА в анамнезе"
DATE_COLS = ["Дата текущего ДКА", "Дата рождения"]
DROP_ALWAYS = ["N пп", "ИБ", LEAKAGE] + DATE_COLS

# Числовые колонки, которые в сыром файле хранятся текстом и требуют разбора.
TEXT_NUMERIC = ["pH при поступлении", "ВЕ при поступлении", "Лактат при поступлении"]


def _split_indices():
    """Индексы train и y по тому же сплиту, что и весь проект."""
    df_clean = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df_clean)
    return df_clean, X_train, y_train


def _raw_frame(train_idx):
    """Сырой фрейм train: приведение текстовых чисел к числам (иначе логрег не
    обучить), без исправления аномалий и без отбора колонок."""
    raw = io.load_raw()
    df = raw.copy()
    for col in TEXT_NUMERIC:
        if col in df.columns:
            df[col] = cleaning._parse_numeric_column(df[col])
    if "Длительность СД (лет)" in df.columns:
        df["Длительность СД (лет)"] = df["Длительность СД (лет)"].map(
            cleaning._parse_diabetes_duration)
    drop = [c for c in DROP_ALWAYS + [config.TARGET_COLUMN] if c in df.columns]
    frame = df.drop(columns=drop).loc[train_idx]
    # Пустые колонки и даты-объекты убираем: они не несут сигнала или не кодируются.
    frame = frame.dropna(axis=1, how="all")
    date_like = [c for c in frame.columns
                 if pd.api.types.is_datetime64_any_dtype(frame[c])]
    frame = frame.drop(columns=date_like)
    quant = [c for c in frame.columns
             if pd.api.types.is_numeric_dtype(frame[c])]
    cat = [c for c in frame.columns if c not in quant]
    return frame, quant, cat


def _stringify(X):
    """Приводит категориальные к строке (NaN -> 'nan'), чтобы one-hot принял
    колонки смешанного типа из сырых данных."""
    return np.asarray(X, dtype=object).astype(str)


def _baseline_logreg(quant, cat):
    """Простая логрег без регуляризации с обязательным минимумом предобработки."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cats = Pipeline([("str", FunctionTransformer(_stringify)),
                     ("enc", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))])
    pre = ColumnTransformer([("num", num, quant), ("cat", cats, cat)],
                            remainder="drop")
    clf = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def _oof(pipe, X, y):
    """Пулированный OOF ROC-AUC и PR-AUC."""
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(pipe, X, y, cv=skf,
                                  method="predict_proba")[:, 1]
    yv = y.to_numpy()
    return roc_auc_score(yv, proba), average_precision_score(yv, proba)


def _finalist_oof():
    """OOF ROC-AUC финалистов из готовой таблицы bootstrap_ci."""
    path = config.TABLES_DIR / "bootstrap_ci.csv"
    return pd.read_csv(path) if path.exists() else None


def run() -> pd.DataFrame:
    df_clean, X_train, y_train = _split_indices()
    feats_nc = features.feature_sets(df_clean)["no_collinear"]
    rows = []

    # Уровень 1: базовая логрег на сырых данных.
    Xr, quant_r, cat_r = _raw_frame(X_train.index)
    auc, pr = _oof(_baseline_logreg(quant_r, cat_r), Xr, y_train)
    rows.append({"уровень": "1. сырые данные, базовая логрег",
                 "признаков": Xr.shape[1], "ROC-AUC (OOF)": round(auc, 3),
                 "PR-AUC (OOF)": round(pr, 3)})

    # Уровень 2: та же базовая логрег на очищенных данных (набор без
    # мультиколлинеарности).
    quant_c, cat_c = features.column_types(df_clean, feats_nc)
    auc, pr = _oof(_baseline_logreg(quant_c, cat_c), X_train[feats_nc], y_train)
    rows.append({"уровень": "2. очищенные данные, базовая логрег",
                 "признаков": len(feats_nc), "ROC-AUC (OOF)": round(auc, 3),
                 "PR-AUC (OOF)": round(pr, 3)})

    table = pd.DataFrame(rows)
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / "baseline_ladder.csv", index=False,
                 encoding="utf-8-sig")

    print(f"Train: {len(y_train)} пациентов, рецидив {int(y_train.sum())} "
          f"({y_train.mean():.1%}). Случайный классификатор: ROC-AUC 0.500, "
          f"PR-AUC {y_train.mean():.3f}.\n")
    print(table.to_string(index=False))

    fin = _finalist_oof()
    if fin is not None:
        print("\nДля сравнения, полный пайплайн (тюнингованные финалисты), OOF:")
        print(fin.to_string(index=False))

    return table


if __name__ == "__main__":
    run()
