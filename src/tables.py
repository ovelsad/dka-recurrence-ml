"""Сводная Таблица 1: характеристики групп по целевой переменной (рецидив).

Для каждого показателя дает описание в двух группах по протоколу, критерий
сравнения, исходное p, скорректированное q (Бенджамини-Хохберг) и отношение
шансов для бинарных признаков. Результат пишет в reports/tables.

Описание количественных согласовано с выбранным критерием сравнения: параметрический
критерий (Стьюдента или Уэлча) применяется только при нормальности обеих групп, тогда
M (SD); иначе непараметрический критерий и Me [Q1; Q3]. Так описание и критерий не
противоречат друг другу в одной строке. Тест и проверка его допущений - в src.stats.
"""

import numpy as np
import pandas as pd

from . import columns, config, io, stats


def _fmt_quant_group(values, normal: bool) -> str:
    """Форматирует количественную сводку для одной группы."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if x.empty:
        return "нет данных"
    if normal:
        return f"{x.mean():.2f} ({x.std(ddof=1):.2f})"
    return f"{x.median():.2f} [{x.quantile(.25):.2f}; {x.quantile(.75):.2f}]"


def build_table_one() -> pd.DataFrame:
    df = io.load_processed()
    roles = columns.classify(df)
    target = roles["target"]
    g = df[target]
    levels = sorted(pd.Series(g).dropna().unique())
    n0 = int((g == levels[0]).sum())
    n1 = int((g == levels[1]).sum())
    col0 = f"Без рецидива (n={n0})"
    col1 = f"Рецидив (n={n1})"

    rows = []
    pvalues = []  # по одному p на показатель, для общей поправки

    # Количественные показатели.
    for col in roles["quantitative"]:
        sub = df[[col, target]].dropna()
        comp = stats.compare_two_groups(sub[col], sub[target])
        # Описание согласуем с критерием: параметрический тест (Стьюдента/Уэлча)
        # выбирается лишь при нормальности обеих групп, тогда M (SD); иначе Me [Q1; Q3].
        normal = comp.test.startswith("t-критерий")
        desc = "M (SD)" if normal else "Me [Q1; Q3]"
        rows.append({
            "Показатель": f"{col}, {desc}",
            col0: _fmt_quant_group(df.loc[g == levels[0], col], normal),
            col1: _fmt_quant_group(df.loc[g == levels[1], col], normal),
            "Критерий": comp.test,
            "p": comp.pvalue,
            "ОШ (95% ДИ)": "",
        })
        pvalues.append(comp.pvalue)

    # Категориальные показатели.
    for col in roles["categorical"]:
        ct = pd.crosstab(df[col], g)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        comp = stats.compare_proportions(ct.to_numpy())
        odds = ""
        if ct.shape == (2, 2):
            orr = stats.odds_ratio(ct.to_numpy())
            odds = (f"{orr['odds_ratio']:.2f} "
                    f"[{orr['ci_low']:.2f}; {orr['ci_high']:.2f}]")

        # Заголовок показателя с тестом.
        rows.append({
            "Показатель": col,
            col0: "", col1: "",
            "Критерий": comp.test,
            "p": comp.pvalue,
            "ОШ (95% ДИ)": odds,
        })
        pvalues.append(comp.pvalue)

        # Доли по категориям считаем от числа известных значений в группе,
        # пропуски в знаменатель не включаем. Так доли согласованы с критерием,
        # который тоже считается по известным значениям.
        d0 = int(ct[levels[0]].sum()) if levels[0] in ct.columns else 0
        d1 = int(ct[levels[1]].sum()) if levels[1] in ct.columns else 0
        for cat in ct.index:
            c0 = int(ct.loc[cat, levels[0]]) if levels[0] in ct.columns else 0
            c1 = int(ct.loc[cat, levels[1]]) if levels[1] in ct.columns else 0
            p0 = f"{c0 / d0 * 100:.1f}%" if d0 else "нет"
            p1 = f"{c1 / d1 * 100:.1f}%" if d1 else "нет"
            rows.append({
                "Показатель": f"    {cat}",
                col0: f"{c0} ({p0})",
                col1: f"{c1} ({p1})",
                "Критерий": "", "p": np.nan, "ОШ (95% ДИ)": "",
            })

    table = pd.DataFrame(rows)

    # Поправка Бенджамини-Хохберга по строкам, где есть p.
    has_p = table["p"].notna()
    q = np.full(len(table), np.nan)
    q[has_p.to_numpy()] = stats.adjust_pvalues(table.loc[has_p, "p"].to_numpy())
    table["q (BH)"] = q

    # Порядок колонок.
    table = table[["Показатель", col0, col1, "Критерий", "p", "q (BH)",
                   "ОШ (95% ДИ)"]]

    _save(table)
    return table


def _save(table: pd.DataFrame) -> None:
    """Сохраняет Таблицу 1 в csv и xlsx."""
    tables_dir = config.TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    rounded = table.copy()
    for c in ("p", "q (BH)"):
        rounded[c] = rounded[c].map(
            lambda v: "" if pd.isna(v) else f"{v:.4f}")
    rounded.to_csv(tables_dir / "table_one.csv", index=False,
                   encoding="utf-8-sig")
    rounded.to_excel(tables_dir / "table_one.xlsx", index=False)


if __name__ == "__main__":
    t = build_table_one()
    print(f"Tablica 1: {len(t)} strok, sohraneno v reports/tables.")
