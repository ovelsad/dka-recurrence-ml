"""Анализ природы пропусков (этап 5, подготовка к импутации).

Оцениваем механизм пропусков: связан ли факт пропуска с целевой переменной и с
другими признаками (это отличает MCAR от MAR), какие показатели пропадают вместе
(совместная пропущенность - признак общей панели обследования). MNAR статистически
не проверяется, отмечаем кандидатов для обсуждения с врачами.

Рисунки идут в reports/figures, отчет в reports/missingness.md.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from . import columns, config, io, stats


def _analysis_columns(df: pd.DataFrame, roles: dict) -> list:
    """Признаки для анализа пропусков (без идентификаторов)."""
    return [c for c in roles["quantitative"] + roles["categorical"] if c in df.columns]


def missingness_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Доля пропусков по колонкам."""
    rows = [{"показатель": c, "пропусков": int(df[c].isna().sum()),
             "доля_%": round(df[c].isna().mean() * 100, 1)} for c in cols]
    return pd.DataFrame(rows).sort_values("доля_%", ascending=False)


def missingness_vs_target(df: pd.DataFrame, cols: list, target: str) -> pd.DataFrame:
    """Связь факта пропуска с целевой: информативна ли пропущенность."""
    g = df[target]
    rows = []
    for c in cols:
        if c == target or df[c].isna().sum() == 0:
            continue
        indicator = df[c].isna().astype(int)
        ct = pd.crosstab(indicator, g)
        if ct.shape == (2, 2) and ct.to_numpy().min() >= 0:
            res = stats.compare_proportions(ct.to_numpy())
            rows.append({"показатель": c, "критерий": res.test, "p": res.pvalue})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q (BH)"] = stats.adjust_pvalues(table["p"].to_numpy())
        table = table.sort_values("p")
    return table


def co_missing_groups(df: pd.DataFrame, cols: list, threshold: float = 0.7) -> list:
    """Группы показателей, которые часто пропадают вместе."""
    miss = df[cols].isna()
    miss = miss.loc[:, miss.any()]
    corr = miss.corr()
    groups = []
    used = set()
    for c in corr.columns:
        if c in used:
            continue
        partners = [o for o in corr.columns
                    if o != c and corr.loc[c, o] >= threshold]
        if partners:
            group = sorted({c, *partners})
            if group not in groups:
                groups.append(group)
                used.update(group)
    return groups


def _save_msno_figures(df: pd.DataFrame, cols: list) -> None:
    """Матрица, тепловая карта и дендрограмма пропусков."""
    sub = df[cols]
    for name, func in (("missing_matrix", msno.matrix),
                       ("missing_heatmap", msno.heatmap),
                       ("missing_dendrogram", msno.dendrogram)):
        ax = func(sub, fontsize=8)
        fig = ax.get_figure()
        fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
        plt.close(fig)


def build_report() -> str:
    df = io.load_processed()
    roles = columns.classify(df)
    target = roles["target"]
    cols = _analysis_columns(df, roles)

    config.ensure_dirs()
    _save_msno_figures(df, cols)

    table = missingness_table(df, cols)
    vs_target = missingness_vs_target(df, cols, target)
    groups = co_missing_groups(df, cols)

    informative = vs_target[vs_target["q (BH)"] < 0.05] if not vs_target.empty \
        else pd.DataFrame()

    lines = ["# Природа пропусков (этап 5)", "",
             "## Доля пропусков", "", table.to_markdown(index=False), "",
             "## Связь пропуска с целевой (информативность)",
             "Если q < 0.05, факт пропуска связан с рецидивом - это не MCAR, "
             "а MAR или MNAR, и простое удаление строк сместит выборку.", "",
             (vs_target.to_markdown(index=False) if not vs_target.empty
              else "нет данных"), "",
             "Информативные пропуски (q < 0.05):",
             (", ".join(informative["показатель"]) if not informative.empty
              else "не найдены"), "",
             "## Совместно пропадающие показатели (общая панель обследования)", ""]
    for grp in groups:
        lines.append(f"- {grp}")
    if not groups:
        lines.append("- выраженных групп нет")
    lines += ["", "## Замечание про MNAR",
              "MNAR статистически не выявляется. Лабораторные, которые берут по "
              "клиническим показаниям (лактат, газы крови, липиды), - кандидаты в "
              "MNAR, обсудить с врачами, когда что назначали."]

    path = config.REPORTS_DIR / "missingness.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    out = build_report()
    print(f"Otchet missingness: {out}")
