"""Обучение и сравнение моделей (этапы 6-7).

Каркас без утечки: train/test split один раз (src.features), весь препроцессинг
(импутация, масштабирование, кодирование, SMOTE) живет в пайплайне и обучается
внутри фолдов стратифицированной кросс-валидации. Импутацию делаем здесь, в
пайплайне, поэтому берем базовый dka_clean с пропусками.

Оси перебора: модель, стратегия импутации, балансировка, набор признаков.
Метрики под клинику и дисбаланс: ROC-AUC, PR-AUC, чувствительность,
специфичность, F1, Brier. По фолдам даем среднее и стандартное отклонение.

Подбор гиперпараметров пока не делаем: цель - ландшафт сравнения на разумных
значениях по умолчанию. Тонкая настройка (включая n_neighbors импутации) - отдельный
следующий шаг.
"""

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC

from . import config, features, io
from .config import RANDOM_SEED
from .manual_impute import ManualImputer

NAN_NATIVE = {"xgb", "lgbm", "catboost"}
NEEDS_SCALING = {"logreg", "svm", "knn"}
SUPPORTS_CLASS_WEIGHT = {"logreg", "rf", "svm", "lgbm", "xgb", "catboost"}


def _estimator(model: str, balancing: str, y, cat_features=None):
    """Создает классификатор с учетом балансировки class_weight."""
    cw = "balanced" if balancing == "class_weight" else None
    if model == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw,
                                  random_state=RANDOM_SEED)
    if model == "rf":
        return RandomForestClassifier(n_estimators=400, class_weight=cw,
                                      random_state=RANDOM_SEED, n_jobs=-1)
    if model == "svm":
        return SVC(probability=True, class_weight=cw, random_state=RANDOM_SEED)
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=7)
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(class_weight=cw, random_state=RANDOM_SEED, verbose=-1)
    if model == "xgb":
        from xgboost import XGBClassifier
        spw = 1.0
        if balancing == "class_weight":
            pos = int((y == 1).sum()); neg = int((y == 0).sum())
            spw = neg / pos if pos else 1.0
        return XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                             random_state=RANDOM_SEED, verbosity=0)
    if model == "catboost":
        from catboost import CatBoostClassifier
        acw = "Balanced" if balancing == "class_weight" else None
        return CatBoostClassifier(iterations=300, verbose=0, random_state=RANDOM_SEED,
                                  auto_class_weights=acw, cat_features=cat_features)
    raise ValueError(model)


def _numeric_imputer(imputation: str):
    if imputation == "median_mode":
        return SimpleImputer(strategy="median")
    if imputation == "knn":
        return KNNImputer(n_neighbors=5)
    if imputation == "mice":
        return IterativeImputer(random_state=RANDOM_SEED, max_iter=15)
    return None  # none


def valid_combo(model: str, imputation: str, balancing: str) -> bool:
    """Отсекает несовместимые сочетания."""
    if imputation == "none":
        if model not in NAN_NATIVE:
            return False
        if balancing == "smote":  # SMOTE не работает с NaN
            return False
    if balancing == "class_weight" and model not in SUPPORTS_CLASS_WEIGHT:
        return False
    if model == "catboost" and balancing == "smote":
        # CatBoost с нативными категориями получает строки, SMOTE их не оверсэмплит.
        return False
    return True


def _categorical_encoder(encoding: str):
    """Кодировщик категориальных: one-hot или target encoding.

    Target encoding (sklearn TargetEncoder) использует внутреннюю перекрестную
    подгонку, поэтому не дает утечки: внутри фолда CV обучается только на train.
    """
    if encoding == "target":
        return TargetEncoder(target_type="binary", random_state=RANDOM_SEED)
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


class _CatBoostPrep(BaseEstimator, TransformerMixin):
    """Подготовка данных для нативной обработки категориальных в CatBoost.

    Числовые импутируем переданным импутером (или оставляем как есть - CatBoost
    обрабатывает пропуски сам), категориальные заполняем отдельной строкой-
    категорией и приводим к строке. Возвращаем DataFrame, чтобы CatBoost получил
    cat_features по именам столбцов. one-hot не делаем: CatBoost кодирует
    категории сам упорядоченными target-статистиками без утечки.
    """

    def __init__(self, quant, cat, num_imputer=None):
        self.quant = quant
        self.cat = cat
        self.num_imputer = num_imputer

    def fit(self, X, y=None):
        self.num_imputer_ = None
        if self.num_imputer is not None and self.quant:
            self.num_imputer_ = clone(self.num_imputer)
            self.num_imputer_.fit(X[self.quant].apply(pd.to_numeric, errors="coerce"))
        return self

    def transform(self, X):
        X = X.copy()
        if self.quant:
            num = X[self.quant].apply(pd.to_numeric, errors="coerce")
            if self.num_imputer_ is not None:
                num = pd.DataFrame(self.num_imputer_.transform(num),
                                   columns=self.quant, index=X.index)
            X[self.quant] = num
        for c in self.cat:
            X[c] = X[c].astype(object).where(X[c].notna(), "nan").astype(str)
        return X[self.quant + self.cat]


def build_pipeline(df, feats, model, imputation, balancing, y, encoding="onehot"):
    """Собирает leakage-safe пайплайн под конкретную конфигурацию."""
    quant, cat = features.column_types(df, feats)

    if model == "catboost":
        # CatBoost обрабатывает категориальные нативно, без one-hot. Числовые
        # импутируем как обычно, категориальные передаем строками с cat_features.
        steps = []
        if imputation == "manual":
            steps.append(("manual", ManualImputer()))
        steps.append(("prep", _CatBoostPrep(quant, cat, _numeric_imputer(imputation))))
        steps.append(("clf", _estimator(model, balancing, y, cat_features=cat)))
        return ImbPipeline(steps)

    num_steps = []
    imp = _numeric_imputer(imputation)
    if imp is not None:
        num_steps.append(("imp", imp))
    if model in NEEDS_SCALING:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", _categorical_encoder(encoding)),
    ])

    pre = ColumnTransformer([("num", num_pipe, quant), ("cat", cat_pipe, cat)],
                            remainder="drop")

    steps = []
    if imputation == "manual":
        # Ручная импутация клиническими правилами до ColumnTransformer: видит все
        # выбранные колонки сразу, обучается на train фолда (без утечки).
        steps.append(("manual", ManualImputer()))
    steps.append(("pre", pre))
    if balancing == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("clf", _estimator(model, balancing, y)))
    return ImbPipeline(steps)


def evaluate_cv(pipeline, X, y, n_splits: int = 5) -> dict:
    """Кросс-валидация с клиническими метриками, среднее и SD по фолдам."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    metrics = {"roc_auc": [], "pr_auc": [], "sens": [], "spec": [], "f1": [],
               "brier": []}
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(Xtr, ytr)
            proba = pipeline.predict_proba(Xva)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
        metrics["roc_auc"].append(roc_auc_score(yva, proba))
        metrics["pr_auc"].append(average_precision_score(yva, proba))
        metrics["sens"].append(tp / (tp + fn) if (tp + fn) else np.nan)
        metrics["spec"].append(tn / (tn + fp) if (tn + fp) else np.nan)
        metrics["f1"].append(f1_score(yva, pred, zero_division=0))
        metrics["brier"].append(brier_score_loss(yva, proba))
    return {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in metrics.items()}


def run_grid(configs: list, dataset: str = "dka_clean") -> pd.DataFrame:
    """Прогоняет список конфигураций, возвращает таблицу метрик CV на train.

    Конфигурация - кортеж (модель, импутация, балансировка, набор) либо
    (модель, импутация, балансировка, набор, кодирование). Если кодирование не
    задано, берем one-hot.
    """
    df = io.load_processed(dataset)
    X_train, _, y_train, _ = features.make_split(df)
    sets = features.feature_sets(df)

    rows = []
    for cfg in configs:
        if len(cfg) == 5:
            model, imputation, balancing, fset, encoding = cfg
        else:
            model, imputation, balancing, fset = cfg
            encoding = "onehot"
        if not valid_combo(model, imputation, balancing):
            continue
        feats = sets[fset]
        pipe = build_pipeline(df, feats, model, imputation, balancing, y_train,
                              encoding)
        res = evaluate_cv(pipe, X_train[feats], y_train)
        row = {"модель": model, "импутация": imputation, "балансировка": balancing,
               "кодирование": encoding, "набор": fset, "n_признаков": len(feats)}
        for k, (m, s) in res.items():
            row[k] = round(m, 3)
            row[f"{k}_sd"] = round(s, 3)
        rows.append(row)
        print(f"{model:9} {imputation:11} {balancing:12} {encoding:7} {fset:12} "
              f"ROC-AUC={row['roc_auc']:.3f}")
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return table


def default_grid() -> list:
    """Грубая карта ландшафта на рабочей импутации KNN плюс честное сравнение
    стратегий импутации.

    Раньше основную сетку гоняли на MICE, а KNN сравнивали только на наборе all.
    Из-за этого импутация спутывалась с набором признаков, и в топе ложно
    доминировал MICE. Теперь основная карта - на KNN (единая с тюнингом, тестом,
    bootstrap и сервисом), а стратегии импутации сравниваем отдельно по всем
    наборам признаков, чтобы сравнение было честным.
    """
    models = ["logreg", "rf", "xgb", "lgbm", "catboost", "svm", "knn"]
    fsets = ["all", "no_collinear", "significant"]
    balancings = ["none", "class_weight", "smote"]

    configs = set()
    # Основная карта ландшафта на рабочей импутации KNN.
    for m in models:
        for fs in fsets:
            for b in balancings:
                configs.add((m, "knn", b, fs))
    # Честное сравнение стратегий импутации: две модели-представителя
    # (линейная и бустинг), все наборы признаков, class_weight.
    for m in ["logreg", "lgbm"]:
        for fs in fsets:
            for imp in ["median_mode", "knn", "mice"]:
                configs.add((m, imp, "class_weight", fs))
    # Базовая линия без импутации для NaN-нативных моделей, все наборы.
    for m in ["lgbm", "xgb", "catboost"]:
        for fs in fsets:
            configs.add((m, "none", "class_weight", fs))
    return sorted(configs)


if __name__ == "__main__":
    table = run_grid(default_grid())
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "model_grid.csv", index=False,
                 encoding="utf-8-sig")
    print("\nЛучшие 5 по ROC-AUC:")
    print(table.head(5).to_string(index=False))
    print(f"\nВсего конфигураций: {len(table)}. Сохранено в reports/tables/model_grid.csv")
