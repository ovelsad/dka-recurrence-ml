"""Рисунки статьи: SHAP моделей сервиса без подписей ноутбука

Модели те же, что в шаге 9.1 ноутбука и в записи data/interim/final_models.json:
логрег Б (набор голосования, C сетки на всем train) и CatBoost с параметрами
Optuna на всем train. Параметры берутся из замороженной записи, модели
обучаются заново на train тем же кодом. Запуск из корня проекта:
python paper/make_figures.py
"""

import ast
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, evaluate, finalist, importance, models, tuning

OUT = Path(__file__).resolve().parent
FIGURES = {
    "логрег Б": ("fig1_shap_logreg_b.png", "Логистическая регрессия Б"),
    "CatBoost": ("fig2_shap_catboost.png", "CatBoost"),
}
# короткие имена колонок ноутбука -> термины статьи
LABELS = {
    "ХБП, С": "Стадия ХБП",
    "ХБП, А": "Альбуминурия",
    "Вид инсулинотерапии на момент ДКА": "Вид инсулинотерапии",
    "Длительность СД": "Длительность диабета",
    "тип СД": "Тип диабета",
    "Невролог": "Полинейропатия",
}


def frozen_params(record: dict, names: list) -> dict:
    """Параметры names модели из замороженной записи (model__x, значение текстом)"""
    return {
        name: ast.literal_eval(record["параметры"][f"model__{name}"]) for name in names
    }


def main() -> None:
    """Обучает модели сервиса на train и сохраняет графики SHAP"""
    frozen = json.loads(finalist.FROZEN_PATH.read_text(encoding="utf-8"))
    X, y = data.model_frame(data.clean(data.load_raw()))
    X_train, _, y_train, _ = evaluate.split(X, y)

    logreg = frozen["модели"]["логрег Б"]
    features = sorted(
        {importance.original_feature(c) for c in logreg["коэффициенты"]},
        key=list(X_train.columns).index,
    )
    params = frozen_params(logreg, ["C", "penalty", "solver"])
    fitted = {"логрег Б": models.logreg(features, **params).fit(X_train, y_train)}
    record = frozen["модели"]["CatBoost"]
    params = frozen_params(record, list(tuning.SPACES["CatBoost"]))
    fitted["CatBoost"] = models.catboost(**params).fit(X_train, y_train)

    # модели должны совпасть с замороженными до теста
    same = finalist.logreg_record(fitted["логрег Б"])["коэффициенты"]
    if same != logreg["коэффициенты"]:
        raise RuntimeError("Логрег Б не совпал с замороженной записью")
    checksum = finalist.prediction_checksum(fitted["CatBoost"], X_train)
    if checksum != record["прогнозы train"]:
        raise RuntimeError("CatBoost не совпал с замороженной записью")

    for name, (file, title) in FIGURES.items():
        values = importance.sum_levels(
            importance.shap_values(name, fitted[name], X_train, X_train)
        )
        fig = importance.plot_shap_beeswarm(values, X_train, title)
        ax = fig.axes[0]
        names = [LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()]
        ax.set_yticklabels(names)
        fig.savefig(OUT / file, dpi=200, bbox_inches="tight")
        print("сохранен", file)


if __name__ == "__main__":
    main()
