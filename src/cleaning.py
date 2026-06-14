"""Очистка исходного датасета по согласованным с заказчиком правилам.

Правила приняты на этапе 1 (см. план и переписку):
- удаляем 10 колонок, пустых на 100 процентов;
- ИБ делаем уникальным ключом: исходный номер сохраняем в "ИБ_исходный",
  повторам и пропускам выдаем новые уникальные номера выше максимума;
- десятичную запятую переводим в точку;
- pH и ВЕ вида "7,2 (6,94)" - берем число до скобки;
- длительность СД: "дебют" -> 0, "N месяц" -> N/12 года;
- количество ДКА: "/" -> пропуск;
- ХБП С оставляем порядковой категорией (1, 2, 3a, 3b, 4, 5);
- натрий менее 100 -> пропуск (ошибка ввода);
- холестерин выше 30 -> делим на 100; ЛПВП выше 5 -> делим на 10;
- креатинин и мочевина равные 0 -> пропуск.

HbA1c около 30 оставляем (врачи подтвердили возможность), но выгружаем номер ИБ
и номер строки в reports/flags_for_doctors.md для перепроверки.

Запуск: python -m src.cleaning (в окружении dka).
"""

import re

import numpy as np
import pandas as pd

from . import config, io

# Текстовые токены, которые означают пропуск.
HIDDEN_NA = {"", " ", "нет данных", "н/д", "na", "nan", "-", "?", "/"}

# Колонка-идентификатор истории болезни.
ID_COL = "ИБ"

# Колонка стадии ХБП, остается порядковой категорией.
CKD_COL = "ХБП, С"
CKD_ORDER = ["1", "2", "3a", "3b", "4", "5"]

# Колонки на удаление с обоснованием (согласовано с заказчиком и врачами).
DROP_COLUMNS = {
    "Количество ДКА в анамнезе": "утечка: по ней врачи ставили целевую переменную",
    "Сутки, на которые произошла нормализация рН": "неинформативная, 245 пропусков",
    "Год текущего ДКА": "неинформативная для прогноза",
    "Применение НМГ (0 - нет, 1 - да)": "квазиконстанта (96.5% одного класса)",
    "Тяжелые гипогликемии в анамнезе (0 - нет, 1 - да)": "квазиконстанта (98.6%)",
    "Количество эпизодов легких гипогликемий в неделю": "квазиконстанта (99.6%)",
    "Целевой HbA1c": "неинформативная",
    "Употребление ПАВ за сутки до ДКА (0-нет, 1 - да)": "квазиконстанта (99.3%)",
    "Данные о смерти": "почти пустая (2 заполненных из 289)",
}


def _to_number(value) -> float:
    """Парсит одно значение в число: запятая в точку, берем часть до скобки."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    if text in HIDDEN_NA:
        return np.nan
    # Берем число до открывающей скобки: "7.2 (6.94)" -> "7.2".
    text = text.split("(")[0].strip()
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else np.nan


def _parse_numeric_column(series: pd.Series) -> pd.Series:
    """Применяет _to_number ко всей колонке."""
    return series.map(_to_number)


def _to_year(value) -> float:
    """Извлекает год: из даты берет .year, из числа или строки - сам год."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "year"):
        return float(value.year)
    match = re.search(r"(19|20)\d{2}", str(value))
    return float(match.group()) if match else np.nan


def _parse_diabetes_duration(value) -> float:
    """Длительность СД в годах: 'дебют' -> 0, 'N месяц' -> N/12."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    if "дебют" in text:
        return 0.0
    if "месяц" in text:
        match = re.search(r"-?\d+(\.\d+)?", text)
        return float(match.group()) / 12 if match else np.nan
    return _to_number(text)


def _normalize_ckd(value) -> object:
    """Нормализует стадию ХБП к строке из CKD_ORDER (кириллица а -> латиница a)."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("а", "a").replace("в", "b")
    # Числовые "1.0" -> "1".
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text if text in CKD_ORDER else text


def reassign_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """Сохраняет исходный ИБ и выдает уникальные номера повторам и пропускам."""
    df = df.copy()
    df["ИБ_исходный"] = df[ID_COL]

    valid = df[ID_COL].dropna()
    next_id = int(valid.max()) + 1 if not valid.empty else 1

    seen: set = set()
    new_ids = []
    for value in df[ID_COL]:
        if pd.isna(value) or value in seen:
            new_ids.append(next_id)
            next_id += 1
        else:
            seen.add(value)
            new_ids.append(value)
    df[ID_COL] = new_ids
    return df


def clean() -> pd.DataFrame:
    df = io.load_raw()
    flags: list[str] = []

    # 1. Удаляем колонки, пустые на 100 процентов.
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    df = df.drop(columns=empty_cols)

    # 2. Уникальный ИБ.
    df = reassign_unique_id(df)

    # 3. Парсинг текстовых числовых колонок.
    for col in ["pH при поступлении", "ВЕ при поступлении", "Лактат при поступлении",
                "Тяжелые гипогликемии в анамнезе (0 - нет, 1 - да)"]:
        if col in df.columns:
            df[col] = _parse_numeric_column(df[col])

    if "Дата текущего ДКА" in df.columns:
        df["Год текущего ДКА"] = df["Дата текущего ДКА"].map(_to_year)
        df = df.drop(columns=["Дата текущего ДКА"])

    if "Длительность СД (лет)" in df.columns:
        df["Длительность СД (лет)"] = df["Длительность СД (лет)"].map(
            _parse_diabetes_duration)

    if "Количество ДКА в анамнезе" in df.columns:
        df["Количество ДКА в анамнезе"] = _parse_numeric_column(
            df["Количество ДКА в анамнезе"])

    if CKD_COL in df.columns:
        df[CKD_COL] = df[CKD_COL].map(_normalize_ckd).astype("category")

    # ХБП А - категория альбуминурии, кодирована числами, но это классы.
    if "ХБП, А" in df.columns:
        df["ХБП, А"] = (
            df["ХБП, А"].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
            .astype("category"))

    # 4. Исправление подозрительных значений.
    if "Натрий при поступлении" in df.columns:
        mask = df["Натрий при поступлении"] < 100
        flags.append(f"Натрий менее 100 -> NaN: {int(mask.sum())} значений.")
        df.loc[mask, "Натрий при поступлении"] = np.nan

    if "Общий холестерин" in df.columns:
        mask = df["Общий холестерин"] > 30
        flags.append(f"Холестерин выше 30 -> делим на 100: {int(mask.sum())}.")
        df.loc[mask, "Общий холестерин"] = df.loc[mask, "Общий холестерин"] / 100

    if "ЛПВП" in df.columns:
        mask = df["ЛПВП"] > 5
        flags.append(f"ЛПВП выше 5 -> делим на 10: {int(mask.sum())}.")
        df.loc[mask, "ЛПВП"] = df.loc[mask, "ЛПВП"] / 10

    for col in ["Креатинин при поступлении", "Мочевина при поступлении"]:
        if col in df.columns:
            mask = df[col] == 0
            flags.append(f"{col} равно 0 -> NaN: {int(mask.sum())}.")
            df.loc[mask, col] = np.nan

    # Степень тяжести ДКА: 0 невозможен, это пропуск.
    if "Степень тяжести ДКА" in df.columns:
        mask = df["Степень тяжести ДКА"] == 0
        flags.append(f"Степень тяжести ДКА равна 0 -> NaN: {int(mask.sum())}.")
        df.loc[mask, "Степень тяжести ДКА"] = np.nan

    # Исправления, выявленные при проверке очищенных данных.
    doctor_flags = []

    if "pH при поступлении" in df.columns:
        mask = df["pH при поступлении"] > 14
        flags.append(f"pH выше 14 -> делим на 10 (потеря точки): {int(mask.sum())}.")
        df.loc[mask, "pH при поступлении"] = df.loc[mask, "pH при поступлении"] / 10

    if "ВЕ при поступлении" in df.columns:
        mask = df["ВЕ при поступлении"].abs() > 40
        flags.append(f"ВЕ по модулю выше 40 -> делим на 10: {int(mask.sum())}.")
        df.loc[mask, "ВЕ при поступлении"] = df.loc[mask, "ВЕ при поступлении"] / 10

    if "Лактат при поступлении" in df.columns:
        mask = df["Лактат при поступлении"] < 0
        flags.append(f"Лактат отрицательный -> NaN: {int(mask.sum())}.")
        for _, row in df[mask].iterrows():
            doctor_flags.append(
                f"- Лактат отрицательный: ИБ исходный {row['ИБ_исходный']}, "
                f"номер строки в датасете (N пп) {row['N пп']}")
        df.loc[mask, "Лактат при поступлении"] = np.nan

    # 5. Флаги для врачей: HbA1c выше 20 (перепроверить первичные данные).
    if "HbA1c" in df.columns:
        high = df[df["HbA1c"] > 20]
        for _, row in high.iterrows():
            doctor_flags.append(
                f"- HbA1c {row['HbA1c']}: ИБ исходный {row['ИБ_исходный']}, "
                f"номер строки в датасете (N пп) {row['N пп']}")

    # 6. Удаление неинформативных колонок и утечки (с обоснованием в логе).
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    for col in to_drop:
        flags.append(f"Удалена колонка '{col}': {DROP_COLUMNS[col]}.")
    df = df.drop(columns=to_drop)

    _write_logs(empty_cols, flags, doctor_flags)
    io.save_processed(df, "dka_clean")
    return df


def _write_logs(empty_cols, flags, doctor_flags) -> None:
    """Пишет обезличенный лог очистки и отдельный файл флагов для врачей."""
    config.ensure_dirs()

    log = ["# Лог очистки (обезличенный)", "",
           f"Удалено пустых колонок: {len(empty_cols)}.", ""]
    log += [f"- {c}" for c in empty_cols]
    log += ["", "## Исправления значений", ""] + [f"- {f}" for f in flags]
    (config.REPORTS_DIR / "cleaning_log.md").write_text(
        "\n".join(log) + "\n", encoding="utf-8")

    doc = ["# Флаги для врачей (содержит номера ИБ, не коммитим)", "",
           "Перепроверить первичные данные по этим записям:", ""]
    doc += doctor_flags if doctor_flags else ["- нет флагов"]
    (config.REPORTS_DIR / "flags_for_doctors.md").write_text(
        "\n".join(doc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cleaned = clean()
    print(f"Ochishcheno: {cleaned.shape[0]} strok, {cleaned.shape[1]} kolonok.")
