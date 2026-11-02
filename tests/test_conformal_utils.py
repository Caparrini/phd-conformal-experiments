"""Tests for dslib.conformal_utils — LAC, Mondrian y métricas de utilidad."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dslib.conformal_utils import (
    bootstrap_auc,
    conformal_threshold,
    coverage_by_class,
    coverage_by_size,
    fast_auc,
    lac_sets,
    mondrian_sets,
    rate_by_decile,
    resplit_coverage,
    triage_table,
    wilson_interval,
)


def _probs(n: int = 6000, seed: int = 0, strength=(1.2, 1.2, 1.2)):
    """Probabilidades de 3 clases, intercambiables entre muestras.

    ``strength`` es la señal de cada clase: con valores distintos, unas clases
    son más difíciles de predecir que otras, como en los datos reales.
    """
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.6, 0.15, 0.25])
    logits = rng.normal(size=(n, 3)) + np.array(strength)[None, :] * np.eye(3)[y]
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True), y


# ---------------------------------------------------------------------------
# umbral
# ---------------------------------------------------------------------------


def test_threshold_is_the_order_statistic():
    s = np.arange(1, 11) / 10  # 0.1 ... 1.0, n = 10
    # k = ceil(11 * 0.9) = 10 -> el máximo.
    assert conformal_threshold(s, 0.10) == pytest.approx(1.0)
    # k = ceil(11 * 0.5) = 6 -> 0.6.
    assert conformal_threshold(s, 0.50) == pytest.approx(0.6)


def test_threshold_is_infinite_without_enough_examples():
    assert conformal_threshold(np.array([0.2, 0.3]), 0.05) == np.inf
    assert conformal_threshold(np.array([]), 0.1) == np.inf


def test_threshold_ignores_float_noise():
    # 19 * 0.9 = 17.1 -> 18; con n+1 = 20 y alpha = 0.1 es 18.0 exacto, no 19.
    s = np.arange(1, 20) / 19  # n = 19
    assert conformal_threshold(s, 0.10) == pytest.approx(18 / 19)


# ---------------------------------------------------------------------------
# conjuntos
# ---------------------------------------------------------------------------


def test_lac_sets_match_mapie():
    from mapie.classification import SplitConformalClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 5))
    y = rng.choice(3, size=4000, p=[0.5, 0.2, 0.3])
    X += y[:, None] * 0.7
    model = LogisticRegression(max_iter=500).fit(X[:1500], y[:1500])
    Xc, yc, Xt = X[1500:3000], y[1500:3000], X[3000:]

    mine, _ = lac_sets(model.predict_proba(Xc), yc, model.predict_proba(Xt), 0.10)
    mapie = SplitConformalClassifier(
        estimator=model, confidence_level=0.90, conformity_score="lac", prefit=True
    ).conformalize(Xc, yc)
    _, theirs = mapie.predict_set(Xt)
    np.testing.assert_array_equal(mine, theirs[:, :, 0])


def test_lac_marginal_coverage_is_about_one_minus_alpha():
    p, y = _probs(n=20000, seed=3)
    sets, _ = lac_sets(p[:10000], y[:10000], p[10000:], 0.10)
    cov = sets[np.arange(10000), y[10000:]].mean()
    assert cov == pytest.approx(0.90, abs=0.015)


def test_mondrian_covers_every_class():
    p, y = _probs(n=30000, seed=4)
    sets, q = mondrian_sets(p[:15000], y[:15000], p[15000:], 0.10)
    cov = coverage_by_class(sets, y[15000:])
    assert q.shape == (3,)
    np.testing.assert_allclose(cov, 0.90, atol=0.03)


def test_mondrian_class_enters_always_when_it_has_too_few_examples():
    p, y = _probs(n=3000, seed=5)
    y = np.where(np.arange(len(y)) < 5, 1, np.where(y == 1, 0, y))  # 5 ejemplos de clase 1
    sets, q = mondrian_sets(p, y, p[:100], 0.05)
    assert q[1] == np.inf
    assert sets[:, 1].all()


def test_mondrian_and_lac_return_expected_shapes():
    p, y = _probs(n=2000)
    s1, q1 = lac_sets(p[:1000], y[:1000], p[1000:], 0.2)
    s2, q2 = mondrian_sets(p[:1000], y[:1000], p[1000:], 0.2)
    assert s1.shape == s2.shape == (1000, 3)
    assert s1.dtype == s2.dtype == bool
    assert np.ndim(q1) == 0 and q2.shape == (3,)


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------


def test_coverage_by_class_and_size():
    sets = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool
    )
    y = np.array([0, 2, 2, 1, 0])
    np.testing.assert_allclose(coverage_by_class(sets, y), [0.5, 1.0, 0.5])
    by_size = coverage_by_size(sets, y).set_index("tamaño")
    assert by_size.loc[1, "cobertura"] == 1.0
    assert by_size.loc[2, "cobertura"] == 0.5  # un acierto y un fallo
    assert by_size.loc[3, "cobertura"] == 1.0
    assert by_size.loc[0, "cobertura"] == 0.0
    assert by_size["préstamos"].sum() == 5


def test_coverage_by_class_nan_when_class_is_absent():
    sets = np.ones((3, 3), dtype=bool)
    assert np.isnan(coverage_by_class(sets, np.array([0, 0, 2]))[1])


def test_triage_table_groups_and_lift():
    names = ["ND", "Dud", "Def"]
    sets = np.array(
        [
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],  # aceptar
            [0, 0, 1],  # escalar
            [1, 1, 0], [1, 1, 1], [0, 0, 0],  # revisar
        ],
        dtype=bool,
    )
    y = np.array([0, 0, 0, 2, 2, 1, 0, 0])
    t = triage_table(sets, y, names).set_index("grupo")
    assert t.loc["Aceptar", "préstamos"] == 4
    assert t.loc["Escalar", "préstamos"] == 1
    assert t.loc["Revisar", "préstamos"] == 3
    assert t.loc["Cartera", "préstamos"] == 8
    assert t.loc["Aceptar", "ND"] == pytest.approx(0.75)
    assert t.loc["Cartera", "Def"] == pytest.approx(0.25)
    # lift = composición del grupo / tasa base: 0.25 / 0.25 en Aceptar para Def.
    assert t.loc["Aceptar", "lift Def"] == pytest.approx(1.0)
    assert t.loc["Cartera", "lift ND"] == pytest.approx(1.0)
    assert t.loc[["Aceptar", "Escalar", "Revisar"], "% cartera"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# re-particiones, AUC, deciles
# ---------------------------------------------------------------------------


def test_resplit_coverage_shape_and_reproducibility():
    p, y = _probs(n=4000)
    a = resplit_coverage(p, y, alpha=0.10, n_splits=3, seed=7)
    b = resplit_coverage(p, y, alpha=0.10, n_splits=3, seed=7)
    assert len(a) == 3 * 2 * 3  # splits x métodos x clases
    assert set(a["método"]) == {"Mondrian", "LAC"}
    assert a["cobertura"].between(0, 1).all()
    np.testing.assert_allclose(a["cobertura"], b["cobertura"])


def test_resplit_mondrian_fixes_per_class_coverage_when_classes_differ():
    # Clases de dificultad desigual: el LAC de un umbral las cubre de forma
    # muy distinta, y Mondrian las deja todas cerca de 1 - alpha.
    p, y = _probs(n=20000, seed=9, strength=(2.0, 0.3, 1.0))
    r = resplit_coverage(p, y, alpha=0.10, n_splits=5, seed=0)
    mean_cov = r.groupby(["método", "clase"])["cobertura"].mean().unstack()
    assert (mean_cov.loc["Mondrian"] - 0.90).abs().max() < 0.03
    assert (mean_cov.loc["LAC"] - 0.90).abs().max() > 0.10


def test_fast_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    s = np.round(rng.normal(size=500) + y, 1)  # con empates
    assert fast_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_fast_auc_is_nan_with_one_class():
    assert np.isnan(fast_auc(np.zeros(5), np.arange(5)))


def test_bootstrap_auc_interval_contains_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    s = rng.normal(size=2000) + 0.8 * y
    auc, lo, hi = bootstrap_auc(y, s, n_boot=50, seed=0)
    assert lo < auc < hi
    assert hi - lo < 0.1
    assert bootstrap_auc(y, s, n_boot=20, seed=3) == bootstrap_auc(y, s, n_boot=20, seed=3)


def test_rate_by_decile_is_monotone_for_an_informative_score():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=20000)
    s = rng.normal(size=20000) + 1.5 * y
    t = rate_by_decile(y, s, k=1, n_bins=10)
    assert len(t) == 10
    assert t["préstamos"].sum() == 20000
    assert t["tasa_real"].is_monotonic_increasing


def test_wilson_interval():
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.9 < hi
    assert 0 <= lo and hi <= 1
    assert np.isnan(wilson_interval(0, 0)[0])
