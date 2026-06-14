"""Онлайн-сервис поддержки решений врача: риск рецидива ДКА (этап 10).

Прототип на Streamlit. Врач выбирает модель и набор признаков, вводит показатели
(или подставляет данные пациента по номеру истории болезни), сервис показывает
калиброванную вероятность повторного эпизода ДКА, относит к группе риска по
выбранному порогу и поясняет вклад признаков.

Модели и пайплайны загружаются из models/service_models.joblib (обучены в
src.service_model). Пропуски в полях заполняет импутер пайплайна, поэтому ответ
выдается даже при неполных данных - но это видно в подсказках.

Запуск: streamlit run service/app.py
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import service_model as sm
from src import features, io
from src.threshold import metrics_at

st.set_page_config(page_title="Риск рецидива ДКА", layout="wide")


@st.cache_resource
def get_bundle():
    return sm.load_bundle()


@st.cache_data
def get_patients():
    """Записи пациентов для подстановки по ИБ: значения, исход, был ли в тесте."""
    df = io.load_processed()
    id_col = "ИБ_исходный" if "ИБ_исходный" in df.columns else "ИБ"
    target_col = [c for c in df.columns if c.startswith("Рецидив")][0]

    _, X_test, _, _ = features.make_split(df)
    test_idx = set(X_test.index)

    records = {}
    for idx, row in df.iterrows():
        ib = row[id_col]
        ib_txt = "без номера" if (ib is None or (isinstance(ib, float) and np.isnan(ib))) \
            else (str(int(ib)) if isinstance(ib, float) and ib.is_integer() else str(ib))
        label = f"ИБ {ib_txt} (строка {idx})"
        outcome = row[target_col]
        records[label] = {
            "values": {c: (None if pd.isna(row[c]) else row[c]) for c in df.columns},
            "outcome": None if pd.isna(outcome) else int(outcome),
            "is_test": idx in test_idx,
        }
    return records


def _value_to_label(spec: dict, value):
    """Подбирает метку варианта категории по сохраненному значению."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return sm.NO_DATA
    for lbl, val in spec["options"]:
        if val is None:
            continue
        if isinstance(val, str):
            if str(value) == val:
                return lbl
        elif float(value) == float(val):
            return lbl
    return sm.NO_DATA


def _apply_patient(records, label, feats, master_spec):
    """Колбэк: заполняет поля ввода значениями выбранного пациента."""
    rec = records[label]
    for col in feats:
        spec = master_spec[col]
        v = rec["values"].get(col)
        if spec["kind"] == "number":
            missing = v is None or (isinstance(v, float) and pd.isna(v))
            st.session_state[f"nd_{col}"] = bool(missing)
            if missing:
                st.session_state[f"in_{col}"] = float(spec["default"])
            else:
                # ограничиваем диапазоном поля, иначе number_input выдаст ошибку
                st.session_state[f"in_{col}"] = float(
                    min(max(float(v), spec["min"]), spec["max"]))
        else:
            st.session_state[f"in_{col}"] = _value_to_label(spec, v)
    st.session_state["loaded_patient"] = label
    st.session_state["patient_outcome"] = rec["outcome"]
    st.session_state["patient_is_test"] = rec["is_test"]


def _init_state(feats, master_spec):
    """Инициализирует значения полей по умолчанию (один раз на ключ)."""
    for col in feats:
        spec = master_spec[col]
        if spec["kind"] == "number":
            st.session_state.setdefault(f"in_{col}", float(spec["default"]))
            st.session_state.setdefault(f"nd_{col}", False)
        else:
            st.session_state.setdefault(f"in_{col}", sm.NO_DATA)


def contributions_figure(contrib: dict, master_spec: dict):
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]))
    labels = [master_spec.get(c, {}).get("label", c) for c, _ in items]
    values = [v for _, v in items]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in values]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(items))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Вклад в риск: вправо - повышает, влево - снижает")
    ax.set_title("Вклад показателей пациента")
    fig.tight_layout()
    return fig


def main():
    bundle = get_bundle()
    master_spec = bundle["master_spec"]

    st.title("Прогноз риска рецидива диабетического кетоацидоза")
    st.caption("Вспомогательный инструмент. Не заменяет клиническое решение врача.")

    with st.sidebar:
        st.header("Модель")
        model = st.selectbox("Алгоритм", list(bundle["models"]),
                             format_func=lambda k: bundle["models"][k])
        fset = st.selectbox("Набор признаков", list(bundle["fsets"]),
                            format_func=lambda k: bundle["fsets"][k])
        entry = bundle["entries"][f"{model}|{fset}"]
        feats = entry["feats"]

        st.header("Порог группы риска")
        st.caption("Порог - вероятность, начиная с которой пациента относим к группе "
                   "риска. Чем ниже порог, тем больше рецидивов поймаем, но больше и "
                   "ложных тревог.")
        default_thr = float(entry["default_threshold"])
        st.caption(f"Порог по умолчанию для этой модели: {default_thr:.2f}. Он выбран "
                   "так, чтобы поймать не менее 85% будущих рецидивов (режим скрининга: "
                   "пропустить пациента группы риска дороже, чем лишний раз "
                   "перепроверить).")
        thr = st.slider("Порог вероятности", 0.05, 0.95, default_thr, 0.01,
                        key=f"thr_{model}_{fset}")
        if abs(thr - default_thr) > 1e-9:
            st.caption(f"Сейчас выбран порог {thr:.2f} (по умолчанию {default_thr:.2f}).")

        m = metrics_at(entry["y_true"], entry["oof"], thr)
        st.markdown(f"**При пороге {thr:.2f} (по контрольной выборке):**")
        st.markdown(
            f"- чувствительность **{m['чувств.']}** - из всех, у кого реально будет "
            f"рецидив, столько модель отнесет к группе риска (выше - реже пропуск)\n"
            f"- специфичность **{m['специф.']}** - из всех, у кого рецидива не будет, "
            f"столько модель верно отнесет к низкому риску (выше - реже ложная тревога)\n"
            f"- PPV **{m['PPV']}** - если модель сказала 'группа риска', с такой "
            f"вероятностью рецидив действительно случится\n"
            f"- NPV **{m['NPV']}** - если модель сказала 'низкий риск', с такой "
            f"вероятностью рецидива действительно не будет")

    _init_state(feats, master_spec)

    # Подстановка пациента по номеру истории болезни.
    records = get_patients()
    with st.expander("Загрузить пациента по номеру истории болезни (ИБ)"):
        label = st.selectbox("Пациент", list(records))
        st.button("Подставить данные пациента", on_click=_apply_patient,
                  args=(records, label, feats, master_spec))
        if st.session_state.get("loaded_patient"):
            out = st.session_state.get("patient_outcome")
            is_test = st.session_state.get("patient_is_test")
            out_txt = {1: "рецидив", 0: "единичный эпизод", None: "нет метки"}[out]
            st.info(f"Загружен {st.session_state['loaded_patient']}. "
                    f"Истинный исход: {out_txt}. "
                    + ("Пациент был в тестовой части валидации."
                       if is_test else
                       "Пациент был в обучающей части валидации."))
            st.caption("Финальная модель обучена на всех пациентах, включая этого, "
                       "поэтому это демонстрация, а не независимая оценка. "
                       "Объективные оценки качества - в карточке модели ниже.")

    st.subheader("Данные пациента")
    values = {}
    cols_ui = st.columns(2)
    for i, col in enumerate(feats):
        spec = master_spec[col]
        with cols_ui[i % 2]:
            if spec["kind"] == "number":
                c1, c2 = st.columns([3, 1])
                nd = c2.checkbox("нет данных", key=f"nd_{col}")
                c1.number_input(spec["label"], min_value=float(spec["min"]),
                                max_value=float(spec["max"]), step=float(spec["step"]),
                                key=f"in_{col}", disabled=nd, help=spec.get("help"))
                values[col] = None if nd else st.session_state[f"in_{col}"]
            else:
                labels = [lbl for lbl, _ in spec["options"]]
                choice = st.selectbox(spec["label"], labels, key=f"in_{col}",
                                      help=spec.get("help"))
                values[col] = dict(spec["options"])[choice]

    if st.button("Рассчитать риск", type="primary"):
        res = sm.predict(bundle, model, fset, values)
        proba = res["proba"]
        band = "повышенный риск рецидива" if proba >= thr else "низкий риск рецидива"
        color = "#C44E52" if proba >= thr else "#55A868"

        left, right = st.columns([1, 1])
        with left:
            st.metric("Вероятность рецидива", f"{proba * 100:.0f}%")
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:{color};"
                f"color:white;font-weight:600;text-align:center'>{band}</div>",
                unsafe_allow_html=True)
            st.caption(f"Порог отнесения: {thr:.2f}")
            n_missing = sum(1 for c in feats if values.get(c) is None)
            if n_missing:
                st.caption(f"Не заполнено полей: {n_missing}. Их значения подставил "
                           "импутер по обучающей выборке - трактуйте такой прогноз "
                           "осторожнее.")
        with right:
            st.pyplot(contributions_figure(res["contributions"], master_spec))

    with st.expander("О модели и ограничениях"):
        mt = entry["metrics"]
        st.markdown(
            f"""
- модель: {bundle['models'][model]}, набор признаков: {bundle['fsets'][fset]}
  ({len(feats)} показателей);
- параметры: {entry['best_params']};
- обучена на {mt['n_train']} размеченных пациентах, доля рецидивов {mt['prevalence']};
- вероятности калиброваны по Платту;
- разделяющая способность по кросс-валидации (ROC-AUC): {mt['oof_roc_auc']};
  на отложенном тесте при валидации ~0.73, доверительный интервал широкий из-за
  малой выборки - модели практически неразличимы;
- калибровка (Brier): {mt['oof_brier']};
- одноцентровая выборка, перед клиническим применением нужна внешняя валидация на
  независимой когорте;
- при чувствительном пороге высокий NPV: модель надежнее отсеивает низкий риск, чем
  подтверждает высокий.
            """)


if __name__ == "__main__":
    main()
