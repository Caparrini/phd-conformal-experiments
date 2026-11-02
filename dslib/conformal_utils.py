"""Utilidades de conformal para clasificación multiclase (LAC y Mondrian).

Todo trabaja sobre matrices de probabilidades ``(n, n_clases)`` y devuelve
conjuntos de predicción como matrices booleanas ``(n, n_clases)``. No depende de
MAPIE, de MLflow ni de Hydra, de modo que se puede repetir una calibración
cientos de veces (re-particiones, barridos de alpha) sin coste.

Puntuación LAC: ``s = 1 - p(clase real)``. La clase entra en el conjunto si
``1 - p(clase) <= q``. El umbral ``q`` es el estadístico de orden
``k = ceil((n + 1) * (1 - alpha))`` de los puntajes de calibración. Es la misma
regla que ``SplitConformalClassifier(conformity_score="lac")`` de MAPIE (el test
lo comprueba).

* ``lac_sets``: un umbral para todos los préstamos. Garantía marginal.
* ``mondrian_sets``: un umbral por clase, calculado solo con los préstamos de
  calibración de esa clase. Garantía por clase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Estadístico de orden ``ceil((n+1)(1-alpha))``; ``inf`` si no hay ejemplos."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    # El round evita que 18.000000000000004 suba un escalón por error de coma flotante.
    rank = int(np.ceil(round((n + 1) * (1 - alpha), 9)))
    return float(s[rank - 1]) if 1 <= rank <= n else np.inf


def lac_sets(
    proba_cal: np.ndarray, y_cal: np.ndarray, proba_new: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    """Conjuntos LAC con un único umbral. Devuelve ``(conjuntos, q)``."""
    y_cal = np.asarray(y_cal)
    scores = 1 - proba_cal[np.arange(len(y_cal)), y_cal]
    q = conformal_threshold(scores, alpha)
    return (1 - proba_new) <= q, q


def mondrian_sets(
    proba_cal: np.ndarray, y_cal: np.ndarray, proba_new: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Conjuntos con un umbral ``q_k`` por clase. Devuelve ``(conjuntos, umbrales)``.

    Si una clase no tiene ejemplos suficientes para el nivel pedido, su umbral
    es infinito y la clase entra siempre en el conjunto.
    """
    y_cal = np.asarray(y_cal)
    n_classes = proba_cal.shape[1]
    q = np.empty(n_classes)
    for k in range(n_classes):
        q[k] = conformal_threshold(1 - proba_cal[y_cal == k, k], alpha)
    return (1 - proba_new) <= q, q


def coverage_by_class(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fracción de préstamos de cada clase real cuyo conjunto la contiene."""
    y = np.asarray(y)
    hit = sets[np.arange(len(y)), y]
    return np.array(
        [hit[y == k].mean() if (y == k).any() else np.nan for k in range(sets.shape[1])]
    )


def coverage_by_size(sets: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Cobertura agrupada por tamaño del conjunto (0, 1, ... n_clases)."""
    y = np.asarray(y)
    size = sets.sum(axis=1)
    hit = sets[np.arange(len(y)), y]
    rows = []
    for s in range(sets.shape[1] + 1):
        mask = size == s
        rows.append(
            {
                "tamaño": s,
                "préstamos": int(mask.sum()),
                "proporción": mask.mean(),
                "cobertura": hit[mask].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def triage_table(
    sets: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    accept_class: int = 0,
    reject_class: int = 2,
) -> pd.DataFrame:
    """Triaje por conjunto: aceptar, escalar o revisar.

    * Aceptar: el conjunto es exactamente ``{accept_class}``.
    * Escalar: el conjunto es exactamente ``{reject_class}``.
    * Revisar: cualquier otro conjunto (vacío o de varias clases).

    Devuelve, por grupo, el número de préstamos, el % de cartera, la
    composición real por clase y el *lift* (composición / tasa base) de cada
    clase. La última fila, ``Cartera``, es la tasa base.
    """
    y = np.asarray(y)
    n, n_classes = len(y), sets.shape[1]
    size = sets.sum(axis=1)
    accept = (size == 1) & sets[:, accept_class]
    reject = (size == 1) & sets[:, reject_class]
    groups = {
        "Aceptar": accept,
        "Escalar": reject,
        "Revisar": ~(accept | reject),
        "Cartera": np.ones(n, dtype=bool),
    }
    base = np.bincount(y, minlength=n_classes) / n
    rows = []
    for name, mask in groups.items():
        comp = (
            np.bincount(y[mask], minlength=n_classes) / mask.sum()
            if mask.any()
            else np.full(n_classes, np.nan)
        )
        row = {"grupo": name, "préstamos": int(mask.sum()), "% cartera": mask.mean()}
        row.update({c: comp[k] for k, c in enumerate(class_names)})
        row.update({f"lift {c}": comp[k] / base[k] for k, c in enumerate(class_names)})
        rows.append(row)
    return pd.DataFrame(rows)


def resplit_coverage(
    proba_pool: np.ndarray,
    y_pool: np.ndarray,
    alpha: float,
    n_splits: int,
    seed: int = 42,
    cal_frac: float = 0.5,
) -> pd.DataFrame:
    """Repite ``n_splits`` veces la partición calibración/test de un mismo conjunto.

    En cada repetición calibra Mondrian y LAC con la parte de calibración y mide
    sobre la parte de test. Devuelve un DataFrame ordenado con una fila por
    ``(split, método, clase)``: ``cobertura`` y ``tamaño medio`` del conjunto.
    """
    y_pool = np.asarray(y_pool)
    n_classes = proba_pool.shape[1]
    idx = np.arange(len(y_pool))
    rows = []
    for i in range(n_splits):
        cal, test = train_test_split(
            idx, train_size=cal_frac, stratify=y_pool, random_state=seed + i
        )
        for method, fn in (("Mondrian", mondrian_sets), ("LAC", lac_sets)):
            sets, _ = fn(proba_pool[cal], y_pool[cal], proba_pool[test], alpha)
            cov = coverage_by_class(sets, y_pool[test])
            size = sets.sum(axis=1).mean()
            for k in range(n_classes):
                rows.append(
                    {
                        "split": i,
                        "método": method,
                        "clase": k,
                        "cobertura": cov[k],
                        "tamaño medio": size,
                    }
                )
    return pd.DataFrame(rows)


def fast_auc(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """AUC por la fórmula de Mann-Whitney con rangos medios (admite empates)."""
    y_bin = np.asarray(y_bin).astype(bool)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y_bin].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(
    y_bin: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 100,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """AUC y su intervalo percentil por *bootstrap*. Devuelve ``(auc, bajo, alto)``."""
    y_bin = np.asarray(y_bin)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boots[b] = fast_auc(y_bin[pick], scores[pick])
    lo, hi = np.nanpercentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return fast_auc(y_bin, scores), float(lo), float(hi)


def rate_by_decile(
    y: np.ndarray, scores: np.ndarray, k: int, n_bins: int = 10
) -> pd.DataFrame:
    """Tasa real de la clase ``k`` por tramo (decil) de su puntuación."""
    df = pd.DataFrame({"hit": np.asarray(y) == k, "score": np.asarray(scores)})
    df["tramo"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop") + 1
    out = df.groupby("tramo").agg(
        préstamos=("hit", "size"),
        puntuación_media=("score", "mean"),
        tasa_real=("hit", "mean"),
    )
    return out.reset_index()


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)
