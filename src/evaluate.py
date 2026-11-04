"""Единая рамка оценки моделей (шаг 2)

Каждая модель проходит одну и ту же процедуру на train:
- внешняя стратифицированная кросс-валидация: 5 повторов по 5 фолдов, одни и те
  же фолды для всех моделей. Если модель сама подбирает гиперпараметры
  (GridSearchCV или обертка над Optuna), подбор идет внутри каждого внешнего
  фолда, и оценка вложенная для того, что подбирается внутри этого объекта.
  Решения, принятые сравнением моделей по этой же CV (импутация, число
  признаков, балансировка), в оценку не вложены, и она немного завышена
- по 25 фолдам считаем ROC-AUC, среднее и разброс идут в правило выбора
  финалиста
- в каждом повторе каждый пациент train предсказан моделью, которая его не
  видела. Это out-of-fold (OOF) предсказания. Метрики, калибровку, порог и
  матрицу ошибок считаем по каждому повтору отдельно и усредняем метрики, а не
  вероятности: среднее по пяти моделям сглаживает прогнозы и приукрашивает
  калибровку. Графики строим по всем повторам, сложенным вместе

Бутстрэп-ДИ считаются при фиксированных OOF-предсказаниях: они отражают разброс
пациентов, но не изменчивость обучения модели, поэтому уже истинной
неопределенности. Порог подбирается по тем же OOF (максимум F1, максимум F2 и
для сравнения Юден), поэтому метрики на нем оптимистичны. Статистических тестов
разности моделей не проводим: событий на train несколько десятков, мощность
таких тестов мала.

Тест в этом модуле не используется, его открываем один раз в конце
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from . import config

N_FOLDS = 5
# Повторы внешней CV: на одном разбиении на 5 фолдов среднее ROC-AUC бейзлайна
# меняется от 0,604 до 0,711 в зависимости от seed разбиения (SD 0,032), этого
# хватает, чтобы перепутать модели. Пять повторов уменьшают этот шум
N_REPEATS = 5
N_BOOTSTRAP = 1000
# 5 квантильных групп: на train из 229 пациентов в группе около 46 человек
CALIBRATION_BINS = 5
# Прогноз 0 или 1 дает бесконечный логит, отступаем от краев на эту величину
LOGIT_EPS = 1e-6
# Пороги для DCA: выше 0,6 врач уже не стал бы ждать модели, чтобы вмешаться
DCA_THRESHOLDS = np.arange(0.05, 0.61, 0.01)
# Цвета моделей в фиксированном порядке, проверены на различимость при
# нарушениях цветового зрения
MODEL_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


def split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Разбиение на train и test по закрепленному списку config.TEST_IDS

    Индекс X и y - N пп. Новые размеченные строки, которых нет в списке, идут в
    train, тест не меняется
    """
    if not X.index.equals(y.index):
        raise ValueError("Индексы X и y не совпадают")
    missing = set(config.TEST_IDS) - set(X.index)
    if missing:
        raise ValueError(f"В данных нет тестовых пациентов N пп {sorted(missing)}")
    in_test = X.index.isin(config.TEST_IDS)
    return X[~in_test], X[in_test], y[~in_test], y[in_test]


def outer_folds() -> RepeatedStratifiedKFold:
    """Внешние фолды: 5 повторов по 5 фолдов, одинаковые для всех моделей"""
    return RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=config.RANDOM_SEED
    )


# ---------------------------------------------------------------- метрики


def share(part: float, rest: float) -> float:
    """Доля part в сумме part + rest, NaN при пустой сумме"""
    return part / (part + rest) if part + rest else np.nan


def youden_threshold(labels, scores) -> float:
    """Порог, максимизирующий чувствительность + специфичность - 1

    Первый порог roc_curve бесконечный (никого не относим к рецидиву), его не
    рассматриваем. Если модель не лучше случайной, получим NaN
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gain = (tpr - fpr)[1:]
    if gain.max() <= 0:
        return np.nan
    return float(thresholds[1:][np.argmax(gain)])


def fbeta_threshold(labels, scores, beta: float) -> float:
    """Порог с максимальной F-мерой по precision (PPV) и recall (чувствительности)

    beta = 1 весит их одинаково, beta = 2 весит recall вдвое сильнее: пропуск
    рецидива обходится дороже ложной тревоги
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]  # последняя точка без порога
    weighted = beta**2 * precision + recall
    fbeta = np.divide(
        (1 + beta**2) * precision * recall,
        weighted,
        out=np.zeros_like(weighted),
        where=weighted > 0,
    )
    return float(thresholds[np.argmax(fbeta)])


# Правила выбора порога для таблицы рабочих точек. Основная точка - максимум F1,
# максимум F2 показываем рядом на равных (решение автора 2026-09-25): при
# ROC-AUC около 0,7 максимум F2 сползает к "помечать почти всех", Юден для сравнения
THRESHOLD_RULES = {
    "максимум F1": lambda labels, scores: fbeta_threshold(labels, scores, beta=1),
    "максимум F2": lambda labels, scores: fbeta_threshold(labels, scores, beta=2),
    "Юден": youden_threshold,
}
MAIN_RULES = ["максимум F1", "максимум F2"]  # точки, которые показываем в выводах


def at_threshold(labels, scores, threshold: float) -> dict:
    """Матрица ошибок и производные метрики при заданном пороге"""
    labels = np.asarray(labels)
    predicted = np.asarray(scores) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "чувствительность": share(tp, fn),
        "специфичность": share(tn, fp),
        "PPV": share(tp, fp),
        "NPV": share(tn, fn),
    }


def calibration(labels, scores) -> dict:
    """Brier, калибровочный наклон и калибровка в среднем

    Brier сравниваем с эталоном: постоянный прогноз, равный доле рецидива,
    дает Brier p(1 - p). Масштабированный Brier = 1 - Brier / эталон: 0 - не
    лучше постоянного прогноза, меньше 0 - хуже него, 1 - идеал.
    Наклон: регрессия исхода на логит прогноза, у идеальной модели 1. Наклон
    меньше 1 значит, что прогнозы слишком крайние (переобучение).
    Калибровка в среднем (calibration-in-the-large): свободный член той же
    регрессии, когда наклон закреплен на 1 (логит идет как offset), у идеальной
    модели 0, меньше 0 - прогнозы завышены. Свободный член из модели с
    оцененным наклоном так читать нельзя, поэтому модели две. При постоянных
    прогнозах или полном разделении регрессия не сходится, тогда возвращаем NaN
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(scores, dtype=float)
    clipped = np.clip(probabilities, LOGIT_EPS, 1 - LOGIT_EPS)
    logit = np.log(clipped / (1 - clipped))
    brier = brier_score_loss(labels, probabilities)
    reference = labels.mean() * (1 - labels.mean())
    result = {"Brier": brier, "Brier масштабированный": 1 - brier / reference}
    try:
        # statsmodels при полном разделении только предупреждает, делаем это ошибкой
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            slope = sm.Logit(labels, sm.add_constant(logit)).fit(disp=0).params[1]
            in_the_large = (
                sm.GLM(
                    labels,
                    np.ones((labels.size, 1)),
                    family=sm.families.Binomial(),
                    offset=logit,
                )
                .fit()
                .params[0]
            )
    except (np.linalg.LinAlgError, ValueError, PerfectSeparationWarning):
        slope, in_the_large = np.nan, np.nan
    result["калибровка: наклон"] = float(slope)
    result["калибровка в среднем"] = float(in_the_large)
    return result


def bootstrap_auc(
    labels, scores, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float, float]:
    """95% ДИ ROC-AUC и PR-AUC бутстрэпом пациентов: (roc_lo, roc_hi, pr_lo, pr_hi)

    scores - вектор прогнозов или матрица повторы x пациенты. Для матрицы в каждом
    ресэмпле метрика считается по каждому повтору на одних и тех же пациентах и
    усредняется. Ресэмплы с одним классом пропускаем
    """
    labels = np.asarray(labels)
    repeats = np.atleast_2d(np.asarray(scores, dtype=float))
    rng = np.random.default_rng(config.RANDOM_SEED)
    rocs, prs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, labels.size, labels.size)
        if np.unique(labels[idx]).size < 2:
            continue
        rocs.append(np.mean([roc_auc_score(labels[idx], r[idx]) for r in repeats]))
        prs.append(
            np.mean([average_precision_score(labels[idx], r[idx]) for r in repeats])
        )
    if not rocs:
        return np.nan, np.nan, np.nan, np.nan
    roc_lo, roc_hi = np.percentile(rocs, [2.5, 97.5])
    pr_lo, pr_hi = np.percentile(prs, [2.5, 97.5])
    return float(roc_lo), float(roc_hi), float(pr_lo), float(pr_hi)


def net_benefit(labels, scores, thresholds=DCA_THRESHOLDS) -> pd.DataFrame:
    """Чистая польза модели и стратегий «лечить всех» и «никого» по порогам

    Чистая польза = TP/n - FP/n * p/(1 - p), где p - порог вероятности, при
    котором врач готов вмешаться
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    n, prevalence = labels.size, labels.mean()
    rows = []
    for p in thresholds:
        odds = p / (1 - p)
        predicted = scores >= p
        tp = (predicted & (labels == 1)).sum() / n
        fp = (predicted & (labels == 0)).sum() / n
        rows.append(
            {
                "порог": p,
                "польза модели": tp - fp * odds,
                "лечить всех": prevalence - (1 - prevalence) * odds,
                "никого": 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- процедура


def cross_validate(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Внешняя повторная CV модели на train: ROC-AUC по фолдам и OOF по повторам

    model - несобранный sklearn-эстиматор, он клонируется на каждом фолде.
    Возвращает словарь: имя, ROC-AUC 25 фолдов, OOF-прогнозы (таблица пациенты x
    повторы), исход, выбранные параметры и пробы Optuna по фолдам. У модели без
    подбора параметры и пробы - списки из None, у GridSearchCV пробы тоже None
    """
    if not X.index.equals(y.index):
        raise ValueError("Индексы X и y не совпадают")
    oof = pd.DataFrame(np.nan, index=y.index, columns=range(N_REPEATS))
    fold_aucs, fold_params, fold_trials = [], [], []
    for fold, (train_idx, valid_idx) in enumerate(outer_folds().split(X, y)):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof.iloc[valid_idx, fold // N_FOLDS] = predicted
        fold_aucs.append(roc_auc_score(y.iloc[valid_idx], predicted))
        # у объекта подбора запоминаем выбранные параметры и пробы Optuna
        fold_params.append(getattr(fitted, "best_params_", None))
        fold_trials.append(getattr(fitted, "trials_", None))
    return {
        "модель": name,
        "ROC-AUC по фолдам": fold_aucs,
        "параметры по фолдам": fold_params,
        "пробы по фолдам": fold_trials,
        "oof": oof,
        "y": y,
    }


def repeat_metrics(labels, scores) -> dict:
    """Метрики одного повтора OOF, не зависящие от порога: AUC и калибровка"""
    return {
        "ROC-AUC": roc_auc_score(labels, scores),
        "PR-AUC": average_precision_score(labels, scores),
        **calibration(labels, scores),
    }


def validate_oof(oof: pd.DataFrame, y: pd.Series) -> None:
    """Не допускает сравнение прогнозов и исходов разных пациентов"""
    if (
        not oof.index.is_unique
        or not y.index.is_unique
        or not oof.index.equals(y.index)
    ):
        raise ValueError(
            "Порядок и уникальность индексов OOF и исхода должны совпадать"
        )
    if list(oof.columns) != list(range(N_REPEATS)):
        raise ValueError("Неверное число или порядок повторов OOF")
    values = oof.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("OOF должен содержать конечные вероятности от 0 до 1")


def summarize(result: dict) -> dict:
    """Строка сводной таблицы: CV по фолдам, метрики OOF как среднее по повторам

    Для ROC-AUC рядом показан размах по повторам. Метрики при пороге - в
    operating_points
    """
    validate_oof(result["oof"], result["y"])
    labels = result["y"].to_numpy()
    repeats = result["oof"].to_numpy().T
    per_repeat = pd.DataFrame([repeat_metrics(labels, scores) for scores in repeats])
    roc_lo, roc_hi, pr_lo, pr_hi = bootstrap_auc(labels, repeats)
    folds = np.asarray(result["ROC-AUC по фолдам"])
    means = per_repeat.mean()
    return {
        "модель": result["модель"],
        "ROC-AUC CV среднее": folds.mean(),
        "ROC-AUC CV SD": folds.std(ddof=1),
        "ROC-AUC OOF": means["ROC-AUC"],
        "ROC-AUC OOF мин": per_repeat["ROC-AUC"].min(),
        "ROC-AUC OOF макс": per_repeat["ROC-AUC"].max(),
        "ROC-AUC ДИ низ": roc_lo,
        "ROC-AUC ДИ верх": roc_hi,
        "PR-AUC OOF": means["PR-AUC"],
        "PR-AUC ДИ низ": pr_lo,
        "PR-AUC ДИ верх": pr_hi,
        "Brier": means["Brier"],
        "Brier эталон": labels.mean() * (1 - labels.mean()),
        "Brier масштабированный": means["Brier масштабированный"],
        "калибровка: наклон": means["калибровка: наклон"],
        "калибровка в среднем": means["калибровка в среднем"],
    }


def operating_points(result: dict) -> pd.DataFrame:
    """Рабочие точки модели по правилам THRESHOLD_RULES

    В каждом повторе порог подбирается по его OOF-прогнозам, метрики на пороге
    усредняются по повторам (поэтому числа матрицы ошибок могут быть дробными).
    Для порога показаны медиана и размах по повторам. Метрики - среднее при
    своем пороге в каждом повторе, а не значения при медианном пороге. Порог
    подобран на тех же прогнозах, по которым считаются метрики, поэтому они
    оптимистичны. Доля помеченных - доля пациентов, отнесенных к рецидиву.
    Повтор, где порог не определен (Юден у модели не лучше случайной), в
    среднее не входит
    """
    labels = result["y"].to_numpy()
    repeats = result["oof"].to_numpy().T
    rows = []
    for rule, choose in THRESHOLD_RULES.items():
        thresholds = np.array([choose(labels, scores) for scores in repeats])
        defined = ~np.isnan(thresholds)
        points = pd.DataFrame(
            [
                at_threshold(labels, s, t)
                for s, t in zip(repeats[defined], thresholds[defined])
            ]
        )
        points["доля помеченных"] = (points["TP"] + points["FP"]) / labels.size
        rows.append(
            {
                "модель": result["модель"],
                "правило": rule,
                "порог": np.median(thresholds[defined]),
                "порог мин": thresholds[defined].min(),
                "порог макс": thresholds[defined].max(),
                "повторов с порогом": int(defined.sum()),
                **points.mean().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def fixed_threshold_points(result: dict, thresholds: dict) -> pd.DataFrame:
    """Метрики OOF при одном заранее заданном пороге во всех повторах

    thresholds - {название: порог}. В отличие от operating_points, порог не
    подбирается в каждом повторе, а один на все: так будет и на тесте. Метрики
    - среднее по повторам
    """
    labels = result["y"].to_numpy()
    repeats = result["oof"].to_numpy().T
    rows = []
    for name, threshold in thresholds.items():
        points = pd.DataFrame([at_threshold(labels, s, threshold) for s in repeats])
        points["доля помеченных"] = (points["TP"] + points["FP"]) / labels.size
        rows.append(
            {
                "модель": result["модель"],
                "порог": name,
                "значение": threshold,
                **points.mean().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def summary_table(results: list) -> pd.DataFrame:
    """Сводная таблица по списку результатов cross_validate, без округления

    Округляем только при показе
    """
    return pd.DataFrame([summarize(r) for r in results])


# ---------------------------------------------------------------- тест (шаг 11)


def test_result(name: str, scores, y: pd.Series) -> dict:
    """Прогнозы на тесте в формате результата cross_validate: один столбец

    Нужен, чтобы строить графики теми же функциями, что и для OOF
    """
    oof = pd.DataFrame({0: np.asarray(scores, dtype=float)}, index=y.index)
    return {"модель": name, "oof": oof, "y": y}


def test_metrics(name: str, scores, y: pd.Series) -> dict:
    """Метрики на тесте, не зависящие от порога: AUC с бутстрэп-ДИ и калибровка"""
    labels = y.to_numpy()
    roc_lo, roc_hi, pr_lo, pr_hi = bootstrap_auc(labels, scores)
    return {
        "модель": name,
        "ROC-AUC": roc_auc_score(labels, scores),
        "ROC-AUC ДИ низ": roc_lo,
        "ROC-AUC ДИ верх": roc_hi,
        "PR-AUC": average_precision_score(labels, scores),
        "PR-AUC ДИ низ": pr_lo,
        "PR-AUC ДИ верх": pr_hi,
        "Brier эталон": labels.mean() * (1 - labels.mean()),
        **calibration(labels, scores),
    }


def test_points(name: str, scores, y: pd.Series, thresholds: dict) -> pd.DataFrame:
    """Матрица ошибок на тесте при замороженных порогах {правило: порог}"""
    labels = y.to_numpy()
    rows = []
    for rule, threshold in thresholds.items():
        point = at_threshold(labels, scores, threshold)
        point["доля помеченных"] = (point["TP"] + point["FP"]) / labels.size
        rows.append({"модель": name, "порог": rule, "значение": threshold, **point})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- графики
# Графики строятся по всем повторам OOF, сложенным вместе: каждый пациент входит
# 5 раз со своим прогнозом из каждого повтора. Так кривая одна на модель, а
# прогнозы не сглажены усреднением. Чистая польза по сложенным повторам равна
# среднему чистых польз по повторам


def model_colors(results: list) -> list:
    """Цвета моделей по порядку, моделей не больше, чем цветов в палитре"""
    if len(results) > len(MODEL_COLORS):
        raise ValueError(f"На одном графике не больше {len(MODEL_COLORS)} моделей")
    return MODEL_COLORS[: len(results)]


def stacked_oof(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Исходы и OOF-прогнозы всех повторов подряд (только для графиков)"""
    oof = result["oof"]
    labels = np.tile(result["y"].to_numpy(), oof.shape[1])
    return labels, oof.to_numpy().T.ravel()


def finish(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Подписи, легенда и сдержанные оси"""
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_curves(results: list, where: str = "OOF на train") -> plt.Figure:
    """ROC- и PR-кривые, where - подпись выборки в заголовке"""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.8))
    for color, result in zip(model_colors(results), results):
        labels, scores = stacked_oof(result)
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=result["модель"])
        ax_pr.plot(recall, precision, color=color, lw=2, label=result["модель"])
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    ax_pr.axhline(results[0]["y"].mean(), color="gray", lw=1, ls="--")
    finish(ax_roc, f"ROC, {where}", "1 - специфичность", "чувствительность")
    finish(ax_pr, f"Precision-recall, {where}", "чувствительность", "PPV")
    fig.tight_layout()
    return fig


def plot_calibration(results: list, where: str = "OOF на train") -> plt.Figure:
    """Калибровочная кривая по квантильным группам прогнозов"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for color, result in zip(model_colors(results), results):
        labels, scores = stacked_oof(result)
        frame = pd.DataFrame({"y": labels, "p": scores})
        frame["группа"] = pd.qcut(frame["p"], CALIBRATION_BINS, duplicates="drop")
        points = frame.groupby("группа", observed=True).mean()
        ax.plot(
            points["p"],
            points["y"],
            marker="o",
            color=color,
            lw=2,
            ms=8,
            label=result["модель"],
        )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    finish(
        ax,
        f"Калибровка, {where}",
        "предсказанная вероятность",
        "наблюдаемая доля рецидива",
    )
    fig.tight_layout()
    return fig


def useful_thresholds(result: dict) -> pd.Series:
    """Интервалы порогов, где чистая польза модели выше обеих стратегий

    Считается по сложенным повторам, как на графике, с шагом сетки
    DCA_THRESHOLDS. Интервалы идут строкой, например "0,22-0,39", отдельные
    интервалы через точку с запятой. Разница бывает тысячными, поэтому
    интервалы читаем вместе с рисунком
    """
    curve = net_benefit(*stacked_oof(result))
    better = (
        curve["польза модели"] > curve[["лечить всех", "никого"]].max(axis=1)
    ).to_numpy()
    thresholds = curve["порог"].to_numpy()
    intervals, start = [], None
    for i, is_better in enumerate(better):
        if is_better and start is None:
            start = thresholds[i]
        last = i == len(better) - 1
        if start is not None and (not is_better or last):
            end = thresholds[i] if is_better else thresholds[i - 1]
            intervals.append(f"{start:.2f}-{end:.2f}".replace(".", ","))
            start = None
    return pd.Series(
        {
            "модель": result["модель"],
            "польза выше обеих стратегий": "; ".join(intervals) or "нет",
        }
    )


def plot_dca(results: list, where: str = "OOF на train") -> plt.Figure:
    """Кривые принятия решений: чистая польза моделей против стратегий «лечить всех» и «никого»"""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for color, result in zip(model_colors(results), results):
        curve = net_benefit(*stacked_oof(result))
        ax.plot(
            curve["порог"],
            curve["польза модели"],
            color=color,
            lw=2,
            label=result["модель"],
        )
    reference = net_benefit(*stacked_oof(results[0]))
    ax.plot(
        reference["порог"],
        reference["лечить всех"],
        color="gray",
        lw=1,
        label="лечить всех",
    )
    ax.axhline(0, color="black", lw=1, label="никого")
    ax.set_ylim(-0.05, results[0]["y"].mean() + 0.05)
    finish(
        ax,
        f"Кривая принятия решений, {where}",
        "порог вероятности",
        "чистая польза",
    )
    fig.tight_layout()
    return fig


def plot_cv_auc(table: pd.DataFrame) -> plt.Figure:
    """ROC-AUC OOF по моделям с 95% бутстрэп-ДИ

    table - сводная таблица summary_table, модели идут сверху вниз в ее порядке.
    Перекрытие ДИ не доказывает равенства моделей и не заменяет тест разности
    """
    table = table.reset_index(drop=True)
    positions = np.arange(len(table))[::-1]
    lower = table["ROC-AUC OOF"] - table["ROC-AUC ДИ низ"]
    upper = table["ROC-AUC ДИ верх"] - table["ROC-AUC OOF"]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(table) + 1.6))
    ax.errorbar(
        table["ROC-AUC OOF"],
        positions,
        xerr=[lower, upper],
        fmt="o",
        color=MODEL_COLORS[0],
        ms=8,
        capsize=4,
    )
    ax.axvline(0.5, color="black", lw=1, label="случайное угадывание")
    ax.set_yticks(positions, table["модель"], fontsize=8)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set(title="ROC-AUC OOF по моделям", xlabel="ROC-AUC OOF и 95% ДИ")
    ax.spines[["top", "right"]].set_visible(False)
    # легенда справа от осей, чтобы не закрывать черты ДИ
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig
