"""Структурный и data-quality обзор датасета перед поколоночным EDA.

Отвечает на стартовые вопросы: одна ли строка на пациента, каков баланс целевой
переменной, какие колонки лежат не в том типе, где скрытые коды пропусков, какие
значения подозрительны по медицинским границам. Результат пишем в
reports/eda_overview.md (UTF-8), чтобы кириллица читалась.

Запуск: python -m src.eda_overview (в окружении dka).
"""

import re

import pandas as pd

from . import config, io

# Грубые физиологические границы для скрининга ошибок ввода.
# Это не клинические нормы, а пределы правдоподобия для поиска опечаток и
# путаницы единиц. Окончательное решение принимает человек.
PLAUSIBLE_RANGES = {
    "Возраст (на текущий момент)": (0, 120),
    "Суточная доза инсулина": (0, 300),
    "HbA1c": (3, 20),
    "Креатинин при поступлении": (20, 2000),
    "Мочевина при поступлении": (0.5, 60),
    "Калий при поступлении": (1.5, 9),
    "Натрий при поступлении": (100, 180),
    "Глюкоза при поступлении": (3, 120),
    "Общий холестерин": (1, 30),
    "ЛПНП": (0, 15),
    "ЛПВП": (0, 5),
    "ТГ": (0, 30),
}

# Подозрительные на скрытые пропуски токены.
HIDDEN_NA_TOKENS = {"", " ", "нет данных", "н/д", "na", "nan", "-", "?"}


def _is_numeric_like(value: str) -> bool:
    """Проверяет, парсится ли строка в число после замены запятой на точку."""
    text = str(value).strip().replace(",", ".")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", text))


def build() -> str:
    df = io.load_raw()
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("# Структурный обзор датасета (этап 1)")
    add()
    add(f"Наблюдений: {len(df)}, колонок: {df.shape[1]}.")
    add()

    # 1. Одна строка на пациента. Проверяем по идентификатору ИБ.
    add("## 1. Одна строка на пациента")
    if "ИБ" in df.columns:
        dup = df["ИБ"].dropna()
        n_dup = int(dup.duplicated().sum())
        add(f"Идентификатор ИБ: заполнен {dup.size}, повторов значений {n_dup}.")
        if n_dup:
            repeated = dup[dup.duplicated(keep=False)].sort_values().unique()[:20]
            add(f"Примеры повторяющихся ИБ: {list(repeated)}")
            add("Вывод: возможны повторные эпизоды у одного пациента, "
                "тогда кросс-валидацию делаем групповой по пациенту.")
        else:
            add("Вывод: повторов нет, одна строка соответствует одному пациенту.")
    add()

    # 2. Целевая переменная.
    add("## 2. Целевая переменная")
    target = config.TARGET_COLUMN
    if target in df.columns:
        counts = df[target].value_counts(dropna=False)
        add(f"Колонка: {target}")
        for value, n in counts.items():
            add(f"- значение {value}: {n} ({n / len(df) * 100:.1f} %)")
        valid = df[target].dropna()
        if valid.nunique() == 2:
            minority = valid.value_counts(normalize=True).min()
            add(f"Баланс классов: меньший класс {minority * 100:.1f} % "
                "среди размеченных строк.")
    add()

    # 3. Колонки, пустые на 100 процентов.
    add("## 3. Полностью пустые колонки")
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        for c in empty_cols:
            add(f"- {c}")
        add("Решение человека: удалить или дозапросить данные.")
    else:
        add("Полностью пустых колонок нет.")
    add()

    # 4. Колонки типа object, которые похожи на числовые.
    add("## 4. Числовые колонки, лежащие в текстовом типе")
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna().astype(str).map(str.strip)
        non_null = non_null[~non_null.str.lower().isin(HIDDEN_NA_TOKENS)]
        if non_null.empty:
            continue
        numeric_mask = non_null.map(_is_numeric_like)
        share_numeric = numeric_mask.mean()
        if share_numeric >= 0.5:
            bad = sorted(set(non_null[~numeric_mask]))[:10]
            add(f"- {col}: числовых {share_numeric * 100:.0f} %, "
                f"мешающие значения: {bad}")
    add()

    # 5. Скрытые коды пропусков в текстовых колонках.
    add("## 5. Скрытые коды пропусков")
    found_hidden = False
    for col in df.columns:
        if df[col].dtype != object:
            continue
        vals = df[col].dropna().astype(str).map(str.strip)
        hidden = vals[vals.str.lower().isin(HIDDEN_NA_TOKENS - {""})]
        empties = int((vals == "").sum())
        if not hidden.empty or empties:
            found_hidden = True
            add(f"- {col}: пустых строк {empties}, "
                f"кодов пропуска {len(hidden)}")
    if not found_hidden:
        add("Явных скрытых кодов пропуска не найдено.")
    add()

    # 6. Подозрительные значения по границам правдоподобия.
    add("## 6. Значения вне границ правдоподобия (кандидаты в ошибки ввода)")
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        out_of_range = numeric[(numeric < low) | (numeric > high)].dropna()
        if not out_of_range.empty:
            sample = sorted(out_of_range.unique())[:10]
            add(f"- {col} (ждем {low}-{high}): {len(out_of_range)} значений, "
                f"примеры {sample}")
    add()

    content = "\n".join(lines) + "\n"
    config.ensure_dirs()
    out_path = config.REPORTS_DIR / "eda_overview.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = build()
    print(f"Obzor zapisan: {path}")
