"""Движок сравнения связанных ROC-кривых: критерий DeLong и парный бутстрэп.

Здесь только переиспользуемые функции сравнения двух AUC на одних и тех же
пациентах (предсказания спарены по пациентам): delong_test - быстрый алгоритм
Sun, Xu (2014) - и paired_bootstrap - бутстрэп разницы AUC. Их использует
src/model_compare.py, где собрана методологическая гипотеза о сложности модели
(все три сложные семейства против логистической регрессии на наборе без
мультиколлинеарности, с поправкой Бенджамини-Хохберга) и попарное сравнение всех
восьми итоговых моделей.

DeLong сравнивает связанные ROC-кривые, метрика ROC инвариантна к калибровке,
поэтому работаем с сырыми вероятностями.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from .config import RANDOM_SEED

N_BOOT = 5000


# --- DeLong для двух связанных ROC-кривых (быстрый алгоритм Sun, Xu, 2014) ---

def _midrank(x):
    """Средние ранги с корректной обработкой совпадений."""
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds, n_pos):
    """AUC и ковариация DeLong для k моделей. preds: [k, n], позитивы первыми."""
    m = n_pos
    n = preds.shape[1] - m
    k = preds.shape[0]
    pos = preds[:, :m]
    neg = preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, p_a, p_b):
    """Двусторонний критерий DeLong для разницы AUC двух связанных моделей."""
    y = np.asarray(y_true)
    order = np.argsort(-y, kind="mergesort")  # позитивы (1) первыми
    n_pos = int(y.sum())
    preds = np.vstack([np.asarray(p_a), np.asarray(p_b)])[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T).ravel()[0])
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * stats.norm.sf(abs(z)))
    return aucs[0], aucs[1], z, p, var


def paired_bootstrap(y_true, p_a, p_b, n_boot=N_BOOT, seed=RANDOM_SEED):
    """Парный бутстрэп разницы AUC: ДИ и двусторонний p по доле смены знака."""
    y = np.asarray(y_true); pa = np.asarray(p_a); pb = np.asarray(p_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        diffs.append(roc_auc_score(ys, pa[idx]) - roc_auc_score(ys, pb[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # двусторонний p: доля бутстрэп-разниц по другую сторону от нуля, удвоенная
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)
