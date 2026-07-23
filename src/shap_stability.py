"""Устойчивость ядра факторов SHAP при бутстрэпе обучающей части.

Замысел. Согласие четырех моделей между собой показывает, что набор факторов не
зависит от выбора алгоритма, но не отвечает на вопрос, зависит ли он от
конкретных 218 пациентов: все четыре модели обучались на одних и тех же данных.
Здесь мы этот вопрос закрываем - пересэмплируем обучающую часть и смотрим, как
часто признак попадает в топ по SHAP.

Процедура одного повтора:
1. бутстрэп-выборка обучающей части (218 наблюдений с возвращением);
2. переобучение всех четырех моделей на этой выборке;
3. расчет SHAP на out-of-bag наблюдениях (те, что в выборку не попали);
4. свертка one-hot уровней к исходным признакам, ранжирование;
5. фиксация состава топ-5, топ-10 и топ-15 у каждой модели.

Что оценка означает и чего не означает:
- гиперпараметры зафиксированы (взяты из tuning_optuna_params.json), поэтому
  изменчивость этапа тюнинга не учтена, и оценка устойчивости завышена, то есть
  служит верхней границей;
- SHAP считается на out-of-bag, чтобы важность не отражала подгонку под ту же
  выборку, на которой модель обучалась;
- три порога вместо одного нужны, чтобы отделить содержательный результат от
  артефакта произвольной границы отсечения;
- бутстрэп говорит только о перевыборке из этой же когорты одного центра.
  Воспроизводимость на данных другого центра он не проверяет.
"""

import json
import warnings

import numpy as np
import pandas as pd
import shap

from . import config, features, io
from . import optuna_tuning as ot
from .config import RANDOM_SEED

MODELS = ["logreg", "rf", "xgb", "catboost"]
FSET = "no_collinear"
TOPS = (5, 10, 15)
N_REPS = 200


def _agg_source(imp_by_name: dict, feats_cat) -> dict:
    """Суммирует важность one-hot уровней обратно к исходному признаку."""
    out = {}
    for name, val in imp_by_name.items():
        matches = [c for c in feats_cat if name.startswith(c + "_")]
        src = max(matches, key=len) if matches else name
        out[src] = out.get(src, 0.0) + float(val)
    return out


def _clean_names(names) -> list:
    return [str(n).replace("num__", "").replace("cat__", "") for n in names]


def _shap_importance(pipe, model: str, X_eval, feats_cat) -> dict:
    """Средний |SHAP| на переданной выборке, свернутый к исходным признакам."""
    if model == "catboost":
        from catboost import Pool

        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        Xt = prep.transform(X_eval)
        sv = clf.get_feature_importance(Pool(Xt, cat_features=list(prep.cat)),
                                        type="ShapValues")
        values = sv[:, :-1]
        names = list(Xt.columns)
    else:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        data = np.asarray(pre.transform(X_eval))
        names = _clean_names(pre.get_feature_names_out())
        if model == "logreg":
            values = shap.LinearExplainer(clf, data).shap_values(data)
        else:
            sv = shap.TreeExplainer(clf)(data)
            values = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    return _agg_source(dict(zip(names, mean_abs)), feats_cat)


def run(n_reps: int = N_REPS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)
    feats = features.feature_sets(df)[FSET]
    _, feats_cat = features.column_types(df, feats)
    params = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))

    n = len(X_train)
    rng = np.random.default_rng(seed)
    rows = []
    skipped = 0

    for rep in range(n_reps):
        idx = rng.choice(n, size=n, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
        # Вырожденные повторы (один класс в выборке или пустой out-of-bag)
        # пропускаем, их доля фиксируется и попадает в отчет.
        if yb.nunique() < 2 or len(oob) < 20:
            skipped += 1
            continue
        Xo = X_train.iloc[oob]

        for model in MODELS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe = ot._build(model, feats, df, yb,
                                     params[f"{model}|{FSET}"])
                    pipe.fit(Xb[feats], yb)
                    imp = _shap_importance(pipe, model, Xo[feats], feats_cat)
            except Exception as exc:  # noqa: BLE001 - повтор пропускаем, не падаем
                print(f"  повтор {rep}, {model}: пропущен ({type(exc).__name__})")
                continue
            order = sorted(imp, key=imp.get, reverse=True)
            for k in TOPS:
                for feat in order[:k]:
                    rows.append({"повтор": rep, "модель": model, "топ": k,
                                 "признак": feat})

        if (rep + 1) % 10 == 0:
            print(f"повторов готово: {rep + 1} из {n_reps}", flush=True)

    long = pd.DataFrame(rows)
    done = long["повтор"].nunique()
    print(f"\nучтено повторов: {done}, пропущено вырожденных: {skipped}")

    # Частота попадания признака в топ-k у каждой модели.
    per_model = (long.groupby(["топ", "признак", "модель"])["повтор"]
                 .nunique().unstack("модель").fillna(0) / done).round(3)
    per_model.columns = [f"частота_{c}" for c in per_model.columns]

    # Частота попадания в ядро: признак в топ-k одновременно у всех моделей.
    core = (long.groupby(["топ", "повтор", "признак"])["модель"].nunique()
            .reset_index())
    core = core[core["модель"] == len(MODELS)]
    core_freq = (core.groupby(["топ", "признак"])["повтор"].nunique() / done).round(3)

    table = per_model.join(core_freq.rename("частота_ядра")).fillna(0.0)
    table["среднее_число_моделей"] = (
        long.groupby(["топ", "признак"])["модель"].count() / done).round(2)
    table = table.reset_index().sort_values(["топ", "частота_ядра"],
                                            ascending=[True, False])

    config.ensure_dirs()
    out = config.TABLES_DIR / "shap_stability.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"сохранено: {out}")
    return table


if __name__ == "__main__":
    result = run()
    print("\n=== топ-10, признаки с частотой ядра выше 0.2 ===")
    sub = result[(result["топ"] == 10) & (result["частота_ядра"] > 0.2)]
    print(sub.to_string(index=False))
