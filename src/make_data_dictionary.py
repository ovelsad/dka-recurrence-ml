"""Строит черновик словаря данных по исходному датасету.

Читает data/raw/dka_bd_289.xlsx и собирает таблицу: имя колонки, тип, число
заполненных и пропусков, число уникальных значений, диапазон или примеры.
Результат пишет в reports/data_dictionary.md. Колонку "смысл" заполняем вручную
вместе с врачебными комментариями.

Запуск: python -m src.make_data_dictionary (в окружении dka).
"""

import pandas as pd

from . import config, io


def _value_hint(series: pd.Series) -> str:
    """Возвращает диапазон для чисел или примеры значений для категорий."""
    non_null = series.dropna()
    if non_null.empty:
        return ""
    if pd.api.types.is_numeric_dtype(series):
        return f"{non_null.min()} ... {non_null.max()}"
    examples = non_null.astype(str).unique()[:5]
    return "; ".join(examples)


def build() -> str:
    df = io.load_raw()
    n_rows = len(df)

    header = [
        "# Словарь данных (черновик)",
        "",
        f"Источник: `data/raw/dka_bd_289.xlsx`. Наблюдений: {n_rows}, "
        f"колонок: {df.shape[1]}.",
        "",
        "Колонку \"смысл\" и единицы измерения заполняем вручную вместе с врачами. "
        "Целевую переменную (факт рецидива) помечаем отдельно.",
        "",
        "| № | колонка | тип | заполнено | пропуски | уникальных | "
        "диапазон или примеры | смысл (заполнить) |",
        "|---|---------|-----|-----------|----------|------------|"
        "----------------------|-------------------|",
    ]

    rows = []
    for i, col in enumerate(df.columns, start=1):
        series = df[col]
        n_filled = int(series.notna().sum())
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        hint = _value_hint(series).replace("|", "/")
        rows.append(
            f"| {i} | {col} | {dtype} | {n_filled} | {n_missing} | "
            f"{n_unique} | {hint} |  |"
        )

    content = "\n".join(header + rows) + "\n"

    config.ensure_dirs()
    out_path = config.REPORTS_DIR / "data_dictionary.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = build()
    print(f"Словарь данных записан: {path}")
