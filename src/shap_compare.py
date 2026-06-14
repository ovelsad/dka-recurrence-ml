"""Сравнение важности признаков по SHAP между финалистами (этап 14).

Для каждого финалиста (логрег, лес, XGBoost, CatBoost) и обоих наборов признаков
(significant 10 и no_collinear 25) считаем SHAP, суммируем вклады one-hot колонок
обратно к исходному признаку и строим топ. Цель - увидеть, какие признаки общие у
моделей, а какие уникальны, и сверить с Таблицей 1.

Дополнительно для CatBoost сравниваем встроенную важность (PredictionValuesChange)
с его же SHAP-ранжированием.

SHAP считаем через `interpret.explanation` (там диспетчеризация по типу модели:
линейный SHAP для логрега, TreeExplainer для леса и XGBoost, встроенные ShapValues
для нативного CatBoost). Гиперпараметры из ноутбука 07.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config, features, interpret, io
from . import optuna_tuning as ot

MODELS = ["logreg", "rf", "xgb", "catboost"]
FSETS = ["significant", "no_collinear"]


def _topn(feats) -> int:
    """Размер топа: 5 для значимого набора (10 признаков), 10 для большого."""
    return 5 if len(feats) <= 10 else 10


def _agg_source(imp_by_name: dict, feats_cat) -> dict:
    """Суммирует важность one-hot уровней обратно к исходному признаку.

    Имена от interpret уже без префиксов num__/cat__. Уровень вида "Признак_1.0"
    сворачиваем к "Признак"; нативные признаки CatBoost (без суффикса уровня)
    остаются как есть.
    """
    out = {}
    for name, val in imp_by_name.items():
        matches = [c for c in feats_cat if name.startswith(c + "_")]
        src = max(matches, key=len) if matches else name
        out[src] = out.get(src, 0.0) + float(val)
    return out


def _shap_source_importance(model, fset, feats_cat) -> dict:
    """Важность по среднему |SHAP|, свернутая к исходным признакам."""
    expl, names, _ = interpret.explanation(model, fset)
    mean_abs = np.abs(expl.values).mean(axis=0)
    return _agg_source(dict(zip(names, mean_abs)), feats_cat)


def _catboost_native(df, X_train, y_train, fset, feats) -> dict:
    """Встроенная важность CatBoost (PredictionValuesChange) на нативной матрице."""
    import json

    from catboost import Pool

    params = json.loads(
        (config.TABLES_DIR / "tuning_optuna_params.json").read_text(encoding="utf-8"))
    pipe = ot._build("catboost", feats, df, y_train, params[f"catboost|{fset}"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train[feats], y_train)
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    Xt = prep.transform(X_train[feats])
    pool = Pool(Xt, cat_features=list(prep.cat))
    imp = clf.get_feature_importance(pool)
    return dict(zip(Xt.columns, imp))


def _plot_top(importance: dict, title: str, path, topn: int) -> list:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    items = items[::-1]
    labels = [c if len(c) <= 38 else c[:36] + ".." for c, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels, vals, color="#4C72B0")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Средний модуль вклада")
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [c for c, _ in items[::-1]]  # топ в порядке убывания


def _overlap_figure(models_top: dict, fset: str, path, topn: int) -> None:
    """Матрица присутствия признака в топ-N у каждой модели."""
    from matplotlib.colors import ListedColormap

    order = []
    for m in MODELS:
        for c in models_top[m]:
            if c not in order:
                order.append(c)
    mat = np.array([[1 if c in models_top[m] else 0 for c in order] for m in MODELS])
    labels = [c if len(c) <= 30 else c[:28] + ".." for c in order]

    cmap = ListedColormap(["#ffffff", "#b6d7a8"])
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(order)), 3.2))
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Присутствие признака в топ-{topn} SHAP по моделям: {fset}",
                 fontsize=10)
    for i in range(len(MODELS)):
        for j in range(len(order)):
            if mat[i, j]:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=9,
                        color="#274e13")
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_catboost_compare(shap_imp: dict, native_imp: dict, fset: str, rho, path,
                           topn: int):
    """Сравнение ранжирования CatBoost: SHAP против встроенной важности (в %)."""
    shap_top = sorted(shap_imp, key=shap_imp.get, reverse=True)[:topn]
    nat_top = sorted(native_imp, key=native_imp.get, reverse=True)[:topn]
    order = list(dict.fromkeys(shap_top + [c for c in nat_top if c not in shap_top]))
    order = order[::-1]

    s_sum = sum(shap_imp.values()) or 1.0
    n_sum = sum(native_imp.values()) or 1.0
    s_vals = [100 * shap_imp.get(c, 0.0) / s_sum for c in order]
    n_vals = [100 * native_imp.get(c, 0.0) / n_sum for c in order]
    labels = [c if len(c) <= 38 else c[:36] + ".." for c in order]

    y = np.arange(len(order))
    h = 0.4
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.5 * len(order))))
    ax.barh(y + h / 2, s_vals, height=h, color="#4C72B0", label="SHAP")
    ax.barh(y - h / 2, n_vals, height=h, color="#DD8452",
            label="встроенная (PredictionValuesChange)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Доля важности, %")
    ax.set_title(f"CatBoost, {fset}: SHAP против встроенной важности "
                 f"(Spearman {rho:.2f})", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    df = io.load_processed()
    X_train, _, y_train, _ = features.make_split(df)

    top_by = {}
    cat_compare = []

    for fset in FSETS:
        feats = features.feature_sets(df)[fset]
        topn = _topn(feats)
        _, feats_cat = features.column_types(df, feats)
        for model in MODELS:
            imp = _shap_source_importance(model, fset, feats_cat)
            safe = f"{model}_{fset}"
            top = _plot_top(imp, f"SHAP топ-{topn}: {model}, {fset}",
                            config.FIGURES_DIR / f"shapcmp_{safe}.png", topn)
            top_by[(model, fset)] = top
            print(f"{model:9} {fset:13} топ: {', '.join(t[:18] for t in top[:5])}")

            if model == "catboost":
                native = _catboost_native(df, X_train, y_train, fset, feats)
                common = [c for c in imp if c in native]
                rho, _ = spearmanr([imp[c] for c in common],
                                   [native[c] for c in common])
                shap_top = sorted(imp, key=imp.get, reverse=True)[:topn]
                nat_top = sorted(native, key=native.get, reverse=True)[:topn]
                cat_compare.append({
                    "набор": fset, "топ": topn,
                    "Spearman SHAP vs встроенная": round(float(rho), 3),
                    "общих в топе": len(set(shap_top) & set(nat_top)),
                })
                _plot_catboost_compare(
                    imp, native, fset, float(rho),
                    config.FIGURES_DIR / f"shap_catboost_compare_{fset}.png", topn)

    lines = ["# Сравнение важности признаков по SHAP между финалистами", "",
             "Размер топа: топ-5 для значимого набора (10 признаков), топ-10 для "
             "набора без мультиколлинеарности (25 признаков). Рисунки "
             "reports/figures/shapcmp_<модель>_<набор>.png.", ""]
    for fset in FSETS:
        feats = features.feature_sets(df)[fset]
        topn = _topn(feats)
        lines += [f"## Набор {fset} (топ-{topn})", ""]
        models_top = {m: top_by[(m, fset)] for m in MODELS}
        _overlap_figure(models_top, fset,
                        config.FIGURES_DIR / f"shap_overlap_{fset}.png", topn)
        common = set.intersection(*[set(v) for v in models_top.values()])
        lines.append(f"Признаки в топ-{topn} у всех моделей: "
                     f"{', '.join(sorted(common)) if common else 'нет'}")
        lines.append("")
        for m in MODELS:
            uniq = [c for c in models_top[m]
                    if all(c not in models_top[o] for o in MODELS if o != m)]
            lines.append(f"- {m}: топ-{topn} {', '.join(models_top[m])}"
                         + (f"; уникальные: {', '.join(uniq)}" if uniq else ""))
        lines.append("")

    lines += ["## CatBoost: SHAP против встроенной важности", "",
              pd.DataFrame(cat_compare).to_markdown(index=False)]

    config.ensure_dirs()
    (config.REPORTS_DIR / "shap_compare.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    rows = [{"модель": m, "набор": f, "топ_признаки": " | ".join(top_by[(m, f)])}
            for f in FSETS for m in MODELS]
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "shap_top_features.csv",
                              index=False, encoding="utf-8-sig")
    print("\nОтчет: reports/shap_compare.md")


if __name__ == "__main__":
    run()
