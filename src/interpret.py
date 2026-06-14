"""SHAP-интерпретация финалистов (этап 10).

Объясняем вклад признаков в предсказание рецидива для всех финалистов: четыре
семейства (логистическая регрессия, случайный лес, XGBoost, CatBoost) на двух
наборах признаков (significant, no_collinear). Глобально: какие признаки в среднем
сильнее двигают риск (beeswarm и столбчатый график средних |SHAP|). Локально:
разбор отдельных пациентов (waterfall) - основа объяснений в сервисе.

Способ расчета SHAP зависит от модели:
- логистическая регрессия - линейный SHAP (LinearExplainer);
- случайный лес и XGBoost - TreeExplainer по преобразованной матрице (после
  импутации и one-hot);
- CatBoost - встроенные ShapValues на нативной матрице с категориями (без one-hot).

Гиперпараметры берем из ноутбука 07 (Optuna). При коррелирующих признаках SHAP
делит вклад между ними, поэтому сверяем выводы с Таблицей 1 и клинической логикой.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config, features, io
from . import optuna_tuning as ot

MODELS = ["logreg", "rf", "xgb", "catboost"]
SETS = ["significant", "no_collinear"]
FINALISTS = [(m, fs) for fs in SETS for m in MODELS]


def _clean_names(names) -> list:
    """Убирает служебные префиксы ColumnTransformer из имен признаков."""
    return [str(n).replace("num__", "").replace("cat__", "") for n in names]


def _load_params() -> dict:
    return json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))


def _fit_pipeline(model: str, fset: str):
    """Собирает финалиста на тюнингованных параметрах и обучает на train."""
    import warnings

    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[fset]
    params = _load_params()[f"{model}|{fset}"]
    pipe = ot._build(model, feats, df, y_train, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train[feats], y_train)
    return pipe, feats, X_train, y_train


def explanation(model: str, fset: str):
    """SHAP-значения финалиста на обучающей матрице.

    Возвращает shap.Explanation (значения для класса рецидива), имена признаков и
    предсказанные вероятности на train (для выбора пациентов под waterfall).
    """
    pipe, feats, X_train, y_train = _fit_pipeline(model, fset)
    n = len(X_train)

    if model == "catboost":
        from catboost import Pool

        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        Xt = prep.transform(X_train[feats])           # DataFrame: количеств. + категории
        cat = list(prep.cat)
        pool = Pool(Xt, cat_features=cat)
        sv = clf.get_feature_importance(pool, type="ShapValues")  # (n, f+1)
        values = sv[:, :-1]
        base = sv[:, -1]
        names = list(Xt.columns)
        # Числовая матрица для раскраски beeswarm: категории заменяем кодами.
        data = Xt.copy()
        for c in cat:
            data[c] = pd.factorize(data[c])[0]
        data = data.to_numpy(dtype=float)
    else:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        data = np.asarray(pre.transform(X_train[feats]))
        names = _clean_names(pre.get_feature_names_out())
        if model == "logreg":
            explainer = shap.LinearExplainer(clf, data)
            values = explainer.shap_values(data)
            ev = explainer.expected_value
            base = np.full(n, float(np.ravel(ev)[0]))
        else:  # rf, xgb
            explainer = shap.TreeExplainer(clf)
            sv = explainer(data)
            values = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
            base = sv.base_values[:, 1] if np.ndim(sv.base_values) == 2 else sv.base_values

    expl = shap.Explanation(values=np.asarray(values), base_values=np.asarray(base),
                            data=data, feature_names=names)
    proba = pipe.predict_proba(X_train[feats])[:, 1]
    return expl, names, proba


def global_plots(expl, names, model: str, fset: str) -> pd.DataFrame:
    """Beeswarm и столбчатый график средней важности, таблица средних |SHAP|."""
    config.ensure_dirs()
    tag = f"{model}_{fset}"

    plt.figure()
    shap.summary_plot(expl.values, features=expl.data, feature_names=names,
                      show=False, max_display=15)
    plt.title(f"SHAP: вклад признаков в риск рецидива ({model}, {fset})", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_beeswarm_{tag}.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(expl.values, features=expl.data, feature_names=names,
                      plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP: средняя важность признаков ({model}, {fset})", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_bar_{tag}.png", bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(expl.values).mean(axis=0)
    return pd.DataFrame({"признак": names, "средний_|SHAP|": mean_abs.round(4)}) \
        .sort_values("средний_|SHAP|", ascending=False).reset_index(drop=True)


def local_examples(expl, proba, model: str, fset: str) -> None:
    """Waterfall для пациента высокого и низкого предсказанного риска."""
    config.ensure_dirs()
    tag = f"{model}_{fset}"
    hi = int(np.argmax(proba))
    lo = int(np.argmin(proba))
    for label, idx in (("high", hi), ("low", lo)):
        plt.figure()
        shap.plots.waterfall(expl[idx], max_display=12, show=False)
        plt.title(f"Пациент {label} риск (p={proba[idx]:.2f}, {model}, {fset})",
                  fontsize=9)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"shap_waterfall_{tag}_{label}.png",
                    bbox_inches="tight")
        plt.close()


def build_report(model: str, fset: str) -> pd.DataFrame:
    """Полный набор SHAP-артефактов одного финалиста."""
    expl, names, proba = explanation(model, fset)
    imp = global_plots(expl, names, model, fset)
    local_examples(expl, proba, model, fset)
    imp.insert(0, "набор", fset)
    imp.insert(0, "модель", model)
    return imp


def build_all() -> pd.DataFrame:
    """SHAP по всем финалистам, сводная таблица топ-важности."""
    config.ensure_dirs()
    parts = []
    for model, fset in FINALISTS:
        imp = build_report(model, fset)
        parts.append(imp)
        print(f"SHAP готов: {model}, {fset}")
    allimp = pd.concat(parts, ignore_index=True)
    allimp.to_csv(config.TABLES_DIR / "shap_importance_finalists.csv", index=False,
                  encoding="utf-8-sig")
    return allimp


if __name__ == "__main__":
    table = build_all()
    print(table.groupby(["модель", "набор"]).head(5).to_string(index=False))
