"""Ручная поколоночная импутация с клиническими правилами (отдельный этап).

Идея: вместо одной автоматической стратегии (медиана, KNN, MICE) на все колонки
заполняем каждый показатель по его смыслу. Часть правил - детерминированные связи
между колонками, они не зависят от данных и не дают утечки:

- возраст манифестации СД = возраст минус длительность СД (и обратные варианты);
- ЛПНП по формуле Фридвальда: ОХ - ЛПВП - ТГ/2.2 (при ТГ < 4.5 ммоль/л);
- степень тяжести ДКА из pH по порогам ADA: pH >= 7.25 - легкая (1),
  7.0-7.24 - средняя (2), < 7.0 - тяжелая (3).

Остальное заполняем статистикой, и ее обучаем ТОЛЬКО на train (внутри фолдов),
поэтому импутер сделан трансформером с fit/transform и встраивается в модельный
пайплайн первым шагом, до ColumnTransformer:

- креатинин - медианой по полу (норма зависит от пола);
- острые и связанные с тяжестью показатели (HbA1c, pH, BE, лактат, калий, натрий) -
  медианой внутри группы по степени тяжести ДКА (решение врача: тяжелее эпизод -
  хуже электролиты, ацидоз и хронический контроль);
- прочие количественные - общей медианой train;
- категориальные - модой, кроме информативных пропусков (см. ниже).

Цель этапа - сравнить качество моделей на ручной импутации с автоматическими
стратегиями тем же leak-free механизмом.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import columns, config, io

# Точные имена колонок очищенного датасета.
C_AGE = "Возраст (на текущий момент)"
C_DUR = "Длительность СД (лет)"
C_ONSET = "Возраст манифестации СД"
C_CREAT = "Креатинин при поступлении"
C_SEX = "Пол (0 - Ж, 1 - М)"
C_HBA1C = "HbA1c"
C_TYPE = "тип СД (1-1, 2-2, 3 -др)"
C_PH = "pH при поступлении"
C_SEV = "Степень тяжести ДКА"
C_BE = "ВЕ при поступлении"
C_TC = "Общий холестерин"
C_HDL = "ЛПВП"
C_TG = "ТГ"
C_LDL = "ЛПНП"
C_LAC = "Лактат при поступлении"
C_K = "Калий при поступлении"
C_NA = "Натрий при поступлении"
C_RETINO = "Ретинопатия (0 - нет, 1 - непролиферативная, 2 - препролиферативная, 3 - пролиферативная)"
C_CKD_C = "ХБП, С"
C_CKD_A = "ХБП, А"
C_NEURO = "Невролог"

# Решения врача по категориальным осложнениям:
SENTINEL = "не обследовано"
# пропуск = осмотр не проводили, делаем это отдельным уровнем (информативный пропуск)
CAT_SENTINEL = [C_RETINO, C_CKD_C, C_CKD_A]
# пропуск = осложнения нет (мода тут увела бы в "есть")
CAT_CONST = {C_NEURO: 0.0}


def _severity_from_ph(ph: float) -> float:
    """Степень тяжести ДКА по порогам ADA."""
    if pd.isna(ph):
        return np.nan
    if ph >= 7.25:
        return 1.0
    if ph >= 7.0:
        return 2.0
    return 3.0


class ManualImputer(BaseEstimator, TransformerMixin):
    """Заполнение пропусков клиническими правилами, без утечки.

    Детерминированные связи применяются как есть, статистики (медианы, моды и
    групповые медианы) обучаются на train в fit и применяются в transform.
    Работает только с теми колонками, что переданы во входной таблице.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.quant_ = [c for c in columns.QUANTITATIVE if c in X.columns]
        self.cat_ = [c for c in columns.CATEGORICAL if c in X.columns]

        num = X[self.quant_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = {c: num[c].median() for c in self.quant_}

        # Групповые медианы для показателей с клинической стратификацией.
        # Острые показатели стратифицируем по степени тяжести ДКА (решение врача):
        # тяжелее эпизод - хуже электролиты, лактат, ацидоз и хронический контроль.
        self.creat_by_sex_ = self._group_median(X, num, C_CREAT, C_SEX)
        self.hba1c_by_sev_ = self._group_median(X, num, C_HBA1C, C_SEV)
        self.ph_by_sev_ = self._group_median(X, num, C_PH, C_SEV)
        self.be_by_sev_ = self._group_median(X, num, C_BE, C_SEV)
        self.lactate_by_sev_ = self._group_median(X, num, C_LAC, C_SEV)
        self.k_by_sev_ = self._group_median(X, num, C_K, C_SEV)
        self.na_by_sev_ = self._group_median(X, num, C_NA, C_SEV)

        self.modes_ = {}
        for c in self.cat_:
            m = X[c].dropna()
            self.modes_[c] = m.mode().iloc[0] if not m.empty else np.nan
        return self

    def _group_median(self, X, num, col, by):
        """Словарь медиан col по группам by (обучается на train)."""
        if col not in X.columns or by not in X.columns:
            return {}
        tmp = pd.DataFrame({by: X[by].values, col: num[col].values})
        return tmp.dropna(subset=[col]).groupby(by)[col].median().to_dict()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.quant_:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        self._fill_age(X)
        self._fill_severity_from_ph(X)
        self._fill_group(X, C_PH, C_SEV, self.ph_by_sev_)
        self._fill_group(X, C_BE, C_SEV, self.be_by_sev_)
        self._fill_group(X, C_HBA1C, C_SEV, self.hba1c_by_sev_)
        self._fill_group(X, C_LAC, C_SEV, self.lactate_by_sev_)
        self._fill_group(X, C_K, C_SEV, self.k_by_sev_)
        self._fill_group(X, C_NA, C_SEV, self.na_by_sev_)
        self._fill_group(X, C_CREAT, C_SEX, self.creat_by_sex_)
        self._fill_ldl_friedewald(X)

        # Остаток количественных - общей медианой train.
        for c in self.quant_:
            med = self.medians_.get(c)
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

        # Категориальные: часть - отдельным уровнем "не обследовано", невролог -
        # клиническим нулем, остальные - модой train (решения врача по колонкам).
        for c in self.cat_:
            if c in CAT_SENTINEL:
                # целиком к строке: иначе float и текстовый уровень не сравнимы
                # при кодировании (OneHotEncoder, parquet)
                X[c] = X[c].astype(object).where(X[c].notna(), SENTINEL).astype(str)
            elif c in CAT_CONST:
                X[c] = X[c].fillna(CAT_CONST[c])
            else:
                X[c] = X[c].fillna(self.modes_.get(c))
        return X

    def _fill_age(self, X):
        """Возраст, манифестация и длительность связаны: одно выводим из двух."""
        if not {C_AGE, C_DUR, C_ONSET} <= set(X.columns):
            return
        m = X[C_ONSET].isna() & X[C_AGE].notna() & X[C_DUR].notna()
        X.loc[m, C_ONSET] = X.loc[m, C_AGE] - X.loc[m, C_DUR]
        m = X[C_DUR].isna() & X[C_AGE].notna() & X[C_ONSET].notna()
        X.loc[m, C_DUR] = X.loc[m, C_AGE] - X.loc[m, C_ONSET]
        m = X[C_AGE].isna() & X[C_ONSET].notna() & X[C_DUR].notna()
        X.loc[m, C_AGE] = X.loc[m, C_ONSET] + X.loc[m, C_DUR]

    def _fill_severity_from_ph(self, X):
        """Степень тяжести из pH по клиническим порогам."""
        if not {C_SEV, C_PH} <= set(X.columns):
            return
        m = X[C_SEV].isna() & X[C_PH].notna()
        X.loc[m, C_SEV] = X.loc[m, C_PH].apply(_severity_from_ph)

    def _fill_group(self, X, col, by, table):
        """Заполняет col групповой медианой по by; что осталось - общей медианой."""
        if col not in X.columns or by not in X.columns or not table:
            return
        m = X[col].isna() & X[by].notna()
        X.loc[m, col] = X.loc[m, by].map(table)

    def _fill_ldl_friedewald(self, X):
        """ЛПНП по Фридвальду при ТГ < 4.5 ммоль/л и положительном результате."""
        if not {C_LDL, C_TC, C_HDL, C_TG} <= set(X.columns):
            return
        m = (X[C_LDL].isna() & X[C_TC].notna() & X[C_HDL].notna()
             & X[C_TG].notna() & (X[C_TG] < 4.5))
        ldl = X.loc[m, C_TC] - X.loc[m, C_HDL] - X.loc[m, C_TG] / 2.2
        ldl = ldl.where(ldl > 0)
        X.loc[m, C_LDL] = ldl


# Описание правил для отчета: колонка, правило, обоснование.
DECISIONS = [
    (C_ONSET, "арифметика: возраст - длительность СД",
     "возраст, манифестация и длительность связаны тождеством"),
    (C_DUR, "арифметика: возраст - возраст манифестации",
     "та же связь, выводим недостающее из двух известных"),
    (C_LDL, "формула Фридвальда: ОХ - ЛПВП - ТГ/2.2",
     "стандартный расчет ЛПНП при ТГ < 4.5 ммоль/л"),
    (C_SEV, "из pH: >=7.25 легкая, 7.0-7.24 средняя, <7.0 тяжелая",
     "пороги тяжести ДКА по ADA задаются именно pH"),
    (C_PH, "медиана внутри группы по степени тяжести",
     "pH тесно связан со степенью тяжести"),
    (C_BE, "медиана внутри группы по степени тяжести",
     "дефицит оснований отражает ту же ацидемию, что и pH"),
    (C_HBA1C, "медиана по степени тяжести",
     "решение врача: тяжелее эпизод - обычно хуже хронический контроль"),
    (C_LAC, "медиана по степени тяжести",
     "лактат растет с тяжестью ДКА"),
    (C_K, "медиана по степени тяжести",
     "электролитные сдвиги выражены при тяжелом ДКА"),
    (C_NA, "медиана по степени тяжести",
     "электролитные сдвиги выражены при тяжелом ДКА"),
    (C_CREAT, "медиана по полу",
     "референсные значения креатинина зависят от пола"),
    ("прочие количественные", "общая медиана train", "нет клинической стратификации"),
    (C_RETINO, "отдельный уровень 'не обследовано'",
     "пропуск = осмотр не проводили, это информативно, а не норма"),
    (C_CKD_C, "отдельный уровень 'не обследовано'",
     "пропуск = стадия ХБП не оценивалась"),
    (C_CKD_A, "отдельный уровень 'не обследовано'",
     "пропуск = альбуминурия не оценивалась (самая дырявая категория, 48%)"),
    (C_NEURO, "0 - нет осложнения",
     "мода тут равна 1, она увела бы пропуски в 'есть'"),
    ("прочие категориальные", "мода train",
     "тип СД, пол, вид инсулинотерапии, алкоголь - самым частым значением"),
]


def build_manual_dataset() -> pd.DataFrame:
    """Сохраняет датасет с ручной импутацией (обучение на всех данных).

    Это только для описательного сравнения распределений и для словаря решений.
    Оценку качества моделей делаем отдельно, обучая импутер на train внутри фолдов.
    """
    df = io.load_processed()
    feats = [c for c in columns.QUANTITATIVE + columns.CATEGORICAL if c in df.columns]
    imp = ManualImputer().fit(df[feats])
    filled = df.copy()
    filled[feats] = imp.transform(df[feats])
    io.save_processed(filled, "dka_imputed_manual")
    return filled


def decisions_table() -> pd.DataFrame:
    """Таблица правил с долей пропусков по колонкам."""
    df = io.load_processed()
    rows = []
    for col, rule, why in DECISIONS:
        if col in df.columns:
            share = round(df[col].isna().mean() * 100, 1)
        else:
            share = ""
        rows.append({"колонка": col, "пропуски_%": share, "правило": rule,
                     "обоснование": why})
    return pd.DataFrame(rows)


def compare_quality(models=("logreg", "lgbm"), fset="all") -> pd.DataFrame:
    """Сравнивает качество моделей по стратегиям импутации (leak-free CV).

    Для каждой модели прогоняем median_mode, knn, mice и manual на одном наборе
    признаков с взвешиванием классов. Импутер обучается на train внутри фолдов.
    """
    from . import modeling  # ленивый импорт: modeling импортирует ManualImputer

    strategies = ["median_mode", "knn", "mice", "manual"]
    configs = [(m, imp, "class_weight", fset) for m in models for imp in strategies]
    table = modeling.run_grid(configs)
    return table


def build_report() -> str:
    config.ensure_dirs()
    filled = build_manual_dataset()
    feats = [c for c in columns.QUANTITATIVE + columns.CATEGORICAL if c in filled.columns]
    residual = int(filled[feats].isna().sum().sum())

    decisions = decisions_table()
    quality = compare_quality()

    lines = ["# Ручная поколоночная импутация (отдельный этап)", "",
             "Заполняем каждый показатель по клиническому смыслу, а не одной общей "
             "стратегией. Детерминированные связи между колонками не зависят от "
             "данных, статистики обучаются на train внутри фолдов.", "",
             f"Остаток пропусков после ручной импутации: {residual}.", "",
             "## Правила по колонкам", "",
             decisions.to_markdown(index=False), "",
             "## Качество моделей: ручная против автоматических", "",
             "Стратифицированная кросс-валидация на train, набор признаков all, "
             "взвешивание классов. Импутер обучается на train фолда.", "",
             quality.to_markdown(index=False), "",
             "Смотрим на ROC-AUC и PR-AUC: если ручная импутация не выше "
             "автоматических, клинические правила не дают прироста разделяющей "
             "способности, и можно остаться на простой автоматической стратегии."]
    path = config.REPORTS_DIR / "manual_imputation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality.to_csv(config.TABLES_DIR / "manual_imputation.csv", index=False,
                   encoding="utf-8-sig")
    return str(path)


if __name__ == "__main__":
    filled = build_manual_dataset()
    n_missing = int(filled[[c for c in columns.QUANTITATIVE + columns.CATEGORICAL
                            if c in filled.columns]].isna().sum().sum())
    print(f"Sohranen dka_imputed_manual, ostatok propuskov: {n_missing}")
    print(decisions_table().to_string(index=False))
