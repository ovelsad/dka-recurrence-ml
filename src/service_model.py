"""Финальные модели для онлайн-сервиса (этап 15): обучение и сохранение.

Сервис дает выбор из восьми финалистов - четыре семейства на двух наборах признаков:
- модели: логистическая регрессия, случайный лес, XGBoost, CatBoost;
- наборы: значимые (10) и без мультиколлинеарности (25).

Гиперпараметры берем из ноутбука 07 (Optuna, tuning_optuna_params.json). Для каждого
финалиста: обучаем тюнингованный пайплайн (импутация KNN, one-hot, у CatBoost нативные
категории), по out-of-fold выбираем калибровку (Платт, только если снижает Brier
>=0.005, иначе сырые вероятности) и подбираем рабочий порог под целевую чувствительность
0.75. Вклад признаков: логрег - coef*значение, лес и XGBoost - SHAP TreeExplainer,
CatBoost - нативные ShapValues.

Развертываемые модели обучаем на ВСЕХ размеченных пациентах (273): этап валидации
(вложенная CV + отложенный тест) пройден ранее, финальную модель честно учим на всех
доступных данных. Пороги и калибровку пересчитываем на 273-OOF. Числа честной оценки
показываем в карточке модели.

Запуск обучения: python -m src.service_model
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, features, io
from . import optuna_tuning as ot
from .config import RANDOM_SEED

MODELS_PATH = config.MODELS_DIR / "service_models.joblib"

NO_DATA = "нет данных"

MODELS = {"logreg": "Логистическая регрессия", "rf": "Случайный лес",
          "xgb": "XGBoost", "catboost": "CatBoost"}


def _load_params() -> dict:
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))
FSETS = {"significant": "Значимые (10)", "no_collinear": "Без мультиколлинеарности (25)"}

# Спецификация ввода для всех признаков (метаданные интерфейса). Набор признаков
# выбирает подмножество отсюда. Значения категорий совпадают с очищенными данными.
_QUANT = [
    ("Возраст (на текущий момент)", "Возраст, лет", "лет", 18.0, 90.0, 1.0, 40.0),
    ("Длительность СД (лет)", "Длительность СД, лет", "лет", 0.0, 60.0, 0.5, 5.0),
    ("Возраст манифестации СД", "Возраст манифестации СД, лет", "лет", 0.0, 80.0, 1.0, 25.0),
    ("Суточная доза инсулина", "Суточная доза инсулина, ед/сут", "ед/сут", 0.0, 150.0, 1.0, 40.0),
    ("HbA1c", "HbA1c, %", "%", 4.0, 30.0, 0.1, 9.0),
    ("Креатинин при поступлении", "Креатинин, мкмоль/л", "мкмоль/л", 0.0, 1000.0, 1.0, 80.0),
    ("Мочевина при поступлении", "Мочевина, ммоль/л", "ммоль/л", 0.0, 50.0, 0.1, 6.0),
    ("pH при поступлении", "pH при поступлении", "", 6.5, 7.6, 0.01, 7.30),
    ("ВЕ при поступлении", "Дефицит оснований (BE), ммоль/л", "ммоль/л", -40.0, 10.0, 0.1, -5.0),
    ("Лактат при поступлении", "Лактат, ммоль/л", "ммоль/л", 0.0, 20.0, 0.1, 1.5),
    ("Калий при поступлении", "Калий, ммоль/л", "ммоль/л", 2.0, 9.0, 0.1, 4.0),
    ("Натрий при поступлении", "Натрий, ммоль/л", "ммоль/л", 100.0, 160.0, 1.0, 140.0),
    ("Глюкоза при поступлении", "Глюкоза, ммоль/л", "ммоль/л", 3.0, 100.0, 0.1, 20.0),
    ("Общий холестерин", "Общий холестерин, ммоль/л", "ммоль/л", 1.0, 20.0, 0.1, 4.5),
    ("ЛПНП", "ЛПНП, ммоль/л", "ммоль/л", 0.0, 10.0, 0.1, 2.5),
    ("ЛПВП", "ЛПВП, ммоль/л", "ммоль/л", 0.2, 5.0, 0.1, 1.2),
    ("ТГ", "Триглицериды, ммоль/л", "ммоль/л", 0.3, 25.0, 0.1, 1.5),
]

_CAT = [
    ("тип СД (1-1, 2-2, 3 -др)", "Тип СД",
     [("СД 1 типа", 1.0), ("СД 2 типа", 2.0), ("Другой", 3.0)]),
    ("Пол (0 - Ж, 1 - М)", "Пол", [("Женский", 0.0), ("Мужской", 1.0)]),
    ("Вид инсулинотерапии (1 - ручки, 2 - помпа) на момент ДКА", "Вид инсулинотерапии",
     [("Шприц-ручки", 1.0), ("Помпа", 2.0), ("Нет инсулинотерапии", 0.0)]),
    ("ХБП, С", "ХБП, стадия по СКФ (С)",
     [("С0 (нет)", "0"), ("С1", "1"), ("С2", "2"), ("С3a", "3a"), ("С3б", "3b"),
      ("С3", "3"), ("С4", "4"), ("С5", "5")]),
    ("ХБП, А", "ХБП, альбуминурия (А)",
     [("А0 (нет)", 0.0), ("А1", 1.0), ("А2", 2.0), ("А3", 3.0)]),
    ("Невролог", "Наличие диабетической полинейропатии", [("Нет", 0.0), ("Есть", 1.0)]),
    ("Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)",
     "Ретинопатия", [("Нет", 0.0), ("Непролиферативная", 1.0),
                     ("Препролиферативная", 2.0), ("Пролиферативная", 3.0)]),
    ("Степень тяжести ДКА", "Степень тяжести ДКА",
     [("Легкая", 1.0), ("Средняя", 2.0), ("Тяжелая", 3.0)]),
    ("Алкоголь за сутки до ДКА (0 - нет, 1 - да)", "Алкоголь за сутки до ДКА",
     [("Нет", 0.0), ("Да", 1.0)]),
]


# Пояснения к признакам для подсказок в интерфейсе (значок вопроса у поля).
_HELP = {
    "Возраст (на текущий момент)": "Возраст пациента на момент эпизода ДКА",
    "Длительность СД (лет)": "Стаж сахарного диабета в годах",
    "Возраст манифестации СД": "Возраст, в котором дебютировал диабет",
    "Суточная доза инсулина": "Суммарная суточная доза инсулина, единиц",
    "HbA1c": "Гликированный гемоглобин, контроль гликемии за последние 3 месяца",
    "Креатинин при поступлении": "Креатинин крови при поступлении, маркер функции почек",
    "Мочевина при поступлении": "Мочевина крови при поступлении",
    "pH при поступлении": "pH крови при поступлении, отражает тяжесть ацидоза",
    "ВЕ при поступлении": "Дефицит оснований (BE), выраженность метаболического ацидоза",
    "Лактат при поступлении": "Лактат крови при поступлении",
    "Калий при поступлении": "Калий сыворотки при поступлении",
    "Натрий при поступлении": "Натрий сыворотки при поступлении",
    "Глюкоза при поступлении": "Глюкоза крови при поступлении",
    "Общий холестерин": "Общий холестерин крови",
    "ЛПНП": "Липопротеины низкой плотности",
    "ЛПВП": "Липопротеины высокой плотности",
    "ТГ": "Триглицериды крови",
    "тип СД (1-1, 2-2, 3 -др)": "Тип сахарного диабета",
    "Пол (0 - Ж, 1 - М)": "Пол пациента",
    "Вид инсулинотерапии (1 - ручки, 2 - помпа) на момент ДКА":
        "Способ введения инсулина на момент эпизода",
    "ХБП, С": "Стадия хронической болезни почек по СКФ",
    "ХБП, А": "Категория альбуминурии (А1-А3)",
    "Невролог": "Наличие диабетической полинейропатии",
    "Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)":
        "Стадия диабетической ретинопатии",
    "Степень тяжести ДКА": "Степень тяжести эпизода ДКА (по pH крови)",
    "Алкоголь за сутки до ДКА (0 - нет, 1 - да)": "Прием алкоголя за сутки до эпизода",
}


def master_spec() -> dict:
    """Словарь col -> спецификация ввода для всех признаков."""
    spec = {}
    for col, label, unit, lo, hi, step, default in _QUANT:
        spec[col] = {"col": col, "label": label, "kind": "number", "unit": unit,
                     "min": lo, "max": hi, "step": step, "default": default,
                     "help": _HELP.get(col)}
    for col, label, options in _CAT:
        spec[col] = {"col": col, "label": label, "kind": "category",
                     "options": options + [(NO_DATA, None)], "help": _HELP.get(col)}
    return spec


def build_input_frame(values: dict, feats: list) -> pd.DataFrame:
    """Однострочный датафрейм по выбранному набору признаков; пропуски остаются NaN."""
    row = {c: (np.nan if values.get(c) is None else values.get(c)) for c in feats}
    return pd.DataFrame([row], columns=feats)


def _aggregate(names, vals, feats_cat) -> dict:
    """Суммирует вклад one-hot обратно к исходному признаку."""
    contrib = {}
    for name, val in zip(names, vals):
        if name.startswith("num__"):
            src = name[len("num__"):]
        elif name.startswith("cat__"):
            rest = name[len("cat__"):]
            matches = [c for c in feats_cat if rest.startswith(c + "_")]
            src = max(matches, key=len) if matches else rest
        else:
            src = name
        contrib[src] = contrib.get(src, 0.0) + float(val)
    return contrib


def _contributions(entry: dict, X_row: pd.DataFrame) -> dict:
    """Вклад признаков: логрег - coef*значение, лес/XGBoost - SHAP, CatBoost - нативный SHAP."""
    raw = entry["raw_pipe"]
    model = entry["model"]
    if model == "catboost":
        # Нативный путь: шаг prep (не pre), ShapValues по Pool с категориями.
        from catboost import Pool

        prep = raw.named_steps["prep"]
        clf = raw.named_steps["clf"]
        Xt = prep.transform(X_row)
        pool = Pool(Xt, cat_features=list(prep.cat))
        sv = clf.get_feature_importance(pool, type="ShapValues")  # (1, f+1)
        return {c: float(v) for c, v in zip(Xt.columns, sv[0, :-1])}

    pre = raw.named_steps["pre"]
    clf = raw.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    Xt = pre.transform(X_row)
    if model == "logreg":
        vals = clf.coef_[0] * np.asarray(Xt).ravel()
    else:  # rf, xgb
        sv = shap.TreeExplainer(clf)(np.asarray(Xt))
        v = sv.values
        v = v[:, :, 1] if getattr(v, "ndim", 2) == 3 else v
        vals = np.asarray(v)[0]
    return _aggregate(names, vals, entry["feats_cat"])


def _train_one(df, X, y, model: str, fset: str) -> dict:
    """Обучает финалиста на всех размеченных, выбирает калибровку и порог по OOF."""
    from .threshold import select_thresholds, metrics_at

    feats = features.feature_sets(df)[fset]
    _, feats_cat = features.column_types(df, feats)
    params = _load_params()[f"{model}|{fset}"]
    y_arr = y.to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # OOF сырых и калиброванных вероятностей для выбора режима калибровки.
        raw_oof = cross_val_predict(ot._build(model, feats, df, y, params),
                                    X[feats], y, cv=skf, method="predict_proba")[:, 1]
        platt_oof = cross_val_predict(
            CalibratedClassifierCV(ot._build(model, feats, df, y, params),
                                   method="sigmoid", cv=5),
            X[feats], y, cv=skf, method="predict_proba")[:, 1]
        use_platt = (brier_score_loss(y_arr, platt_oof)
                     <= brier_score_loss(y_arr, raw_oof) - 0.005)
        oof = platt_oof if use_platt else raw_oof

        # Отдельный сырой пайплайн для объяснений вкладов (coef/SHAP).
        raw_pipe = ot._build(model, feats, df, y, params)
        raw_pipe.fit(X[feats], y)
        if use_platt:
            deployed = CalibratedClassifierCV(ot._build(model, feats, df, y, params),
                                              method="sigmoid", cv=5)
            deployed.fit(X[feats], y)
        else:
            deployed = raw_pipe

    thresholds = select_thresholds(y_arr, oof, sens_target=0.75)
    default_thr = thresholds["чувств.>=0.75"]
    metrics = {
        "oof_roc_auc": round(float(roc_auc_score(y_arr, oof)), 3),
        "oof_brier": round(float(brier_score_loss(y_arr, oof)), 3),
        "n_train": int(len(y)),
        "prevalence": round(float(y_arr.mean()), 3),
        "at_default": metrics_at(y_arr, oof, default_thr),
    }
    return {
        "calibrated": deployed, "raw_pipe": raw_pipe, "model": model,
        "kind": "linear" if model == "logreg" else "tree",
        "feats": feats, "feats_cat": feats_cat,
        "thresholds": thresholds, "default_threshold": float(default_thr),
        "calibration": "sigmoid" if use_platt else "none",
        "metrics": metrics, "best_params": params,
        "oof": oof.astype(float), "y_true": y_arr.astype(int),
    }


def train_and_save() -> dict:
    """Обучает все комбинации модель x набор и сохраняет один бандл."""
    config.ensure_dirs()
    df = io.load_processed()
    X, y = features.split_xy(df)

    entries = {}
    for model in MODELS:
        for fset in FSETS:
            key = f"{model}|{fset}"
            print(f"обучаю {key} ...")
            entries[key] = _train_one(df, X, y, model, fset)
            print(f"  ROC-AUC(OOF)={entries[key]['metrics']['oof_roc_auc']} "
                  f"порог={entries[key]['default_threshold']:.3f}")

    bundle = {
        "entries": entries,
        "master_spec": master_spec(),
        "models": MODELS,
        "fsets": FSETS,
        "feature_sets": {fs: features.feature_sets(df)[fs] for fs in FSETS},
        "target": config.TARGET_COLUMN,
    }
    joblib.dump(bundle, MODELS_PATH)
    return bundle


def load_bundle() -> dict:
    """Загружает сохраненные модели сервиса."""
    return joblib.load(MODELS_PATH)


def predict(bundle: dict, model: str, fset: str, values: dict) -> dict:
    """Вероятность рецидива и вклад признаков для выбранной комбинации."""
    entry = bundle["entries"][f"{model}|{fset}"]
    X_row = build_input_frame(values, entry["feats"])
    proba = float(entry["calibrated"].predict_proba(X_row)[:, 1][0])
    return {"proba": proba, "contributions": _contributions(entry, X_row)}


if __name__ == "__main__":
    b = train_and_save()
    print("\nСохранено:", MODELS_PATH)
    for key, e in b["entries"].items():
        print(f"{key:28} ROC-AUC(OOF)={e['metrics']['oof_roc_auc']} "
              f"параметры={e['best_params']}")
