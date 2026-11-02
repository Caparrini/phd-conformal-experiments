"""Tests for dslib.alpha_sweep — alpha-sensitivity mechanics on binary p-values."""

from __future__ import annotations

import numpy as np
import pytest

from dslib.alpha_sweep import (
    check,
    empty_set_threshold,
    marginal_precision,
    sets_from_p_values,
    sweep,
    table_for_paper,
    to_latex,
    transitions,
)


def _synthetic_p_values(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    base = rng.beta(2, 2, size=n)
    p1 = np.where(y == 1, 1 - base, base)
    p0 = np.clip(1 - p1 + rng.normal(0, 0.05, size=n), 0.0, 1.0)
    P = np.column_stack([p0, p1])
    return P, y


# ---------------------------------------------------------------------------
# sets_from_p_values
# ---------------------------------------------------------------------------


def test_sets_from_p_values_all_four_states():
    P = np.array(
        [
            [0.8, 0.8],  # both above alpha -> two-set
            [0.8, 0.1],  # only class 0
            [0.1, 0.8],  # only class 1
            [0.1, 0.1],  # both below -> empty
        ]
    )
    sets = sets_from_p_values(P, alpha=0.5)
    assert sets == [{0, 1}, {0}, {1}, set()]


def test_sets_from_p_values_boundary_is_exclusive():
    # p_y > alpha is strict; p_y == alpha must NOT be included.
    P = np.array([[0.5, 0.9]])
    sets = sets_from_p_values(P, alpha=0.5)
    assert sets == [{1}]


# ---------------------------------------------------------------------------
# sweep + check
# ---------------------------------------------------------------------------


def test_sweep_check_passes_on_synthetic_data():
    P, y = _synthetic_p_values()
    alphas = np.round(np.arange(0.01, 0.31, 0.01), 4)
    df = sweep(P, y, alphas)
    check(df, P, y)  # must not raise


def test_sweep_alpha_010_matches_direct_computation():
    P, y = _synthetic_p_values()
    df = sweep(P, y, [0.10])
    row = df.iloc[0]

    sets = sets_from_p_values(P, 0.10)
    sizes = np.array([len(s) for s in sets])
    covered = np.array([y[i] in sets[i] for i in range(len(y))])

    assert row["coverage"] == pytest.approx(np.mean(covered))
    assert row["avg_set_size"] == pytest.approx(np.mean(sizes))
    assert row["TS"] == int(np.sum(sizes == 2))
    assert row["E"] == int(np.sum(sizes == 0))
    assert row["TP"] + row["FP"] + row["TN"] + row["FN"] == int(np.sum(sizes == 1))


def test_sweep_reports_summary_columns_for_plot_coverage_vs_confidence():
    P, y = _synthetic_p_values(n=500)
    df = sweep(P, y, [0.10, 0.20])
    for col in ["Confidence", "Coverage", "One_Class_Correct", "One_Class_Incorrect", "Two_Class_Sets", "No_Sets"]:
        assert col in df.columns
    row = df[df["alpha"] == 0.10].iloc[0]
    assert row["Confidence"] == pytest.approx(0.90)
    assert row["Coverage"] == pytest.approx(row["coverage"])


def test_check_raises_on_broken_outcome_totals():
    P, y = _synthetic_p_values(n=200)
    df = sweep(P, y, [0.10, 0.20])
    df.loc[0, "TP"] = df.loc[0, "TP"] + 1  # break the n_test total
    with pytest.raises(AssertionError):
        check(df, P, y)


def test_check_raises_on_non_monotonic_avg_set_size():
    P, y = _synthetic_p_values(n=200)
    df = sweep(P, y, [0.10, 0.20])
    df.loc[df["alpha"] == 0.20, "avg_set_size"] = df["avg_set_size"].max() + 1
    with pytest.raises(AssertionError):
        check(df, P, y)


# ---------------------------------------------------------------------------
# marginal_precision
# ---------------------------------------------------------------------------


def test_marginal_precision_matches_direct_band_computation():
    P, y = _synthetic_p_values(n=3000)
    alphas = np.round(np.arange(0.01, 0.11, 0.01), 4)
    mp = marginal_precision(P, y, alphas)

    min_p = np.minimum(P[:, 0], P[:, 1])
    singleton_class = np.argmax(P, axis=1)
    edges = np.concatenate(([0.0], alphas))

    for k, row in mp.iterrows():
        mask = (min_p >= edges[k]) & (min_p < edges[k + 1])
        n_entrants = int(np.sum(mask))
        assert row["n_entrants"] == n_entrants
        if n_entrants > 0:
            correct = singleton_class[mask] == y[mask]
            assert row["precision"] == pytest.approx(np.mean(correct))
            assert row["n_correct"] == int(np.sum(correct))


def test_marginal_precision_requires_sorted_alphas():
    P, y = _synthetic_p_values(n=100)
    with pytest.raises(ValueError):
        marginal_precision(P, y, [0.20, 0.10])


# ---------------------------------------------------------------------------
# transitions
# ---------------------------------------------------------------------------


def test_transitions_respects_nesting():
    P, y = _synthetic_p_values(n=5000)
    tr = transitions(P, y, 0.05, 0.20)
    # A sample cannot grow: two_set can go anywhere, singletons can only
    # stay or become empty, empty can only stay empty.
    assert tr.loc["empty", "singleton_0"] == 0
    assert tr.loc["empty", "singleton_1"] == 0
    assert tr.loc["empty", "two_set"] == 0
    assert tr.loc["singleton_0", "two_set"] == 0
    assert tr.loc["singleton_1", "two_set"] == 0
    assert tr.values.sum() == len(y)


def test_transitions_rejects_unordered_alphas():
    P, y = _synthetic_p_values(n=100)
    with pytest.raises(ValueError):
        transitions(P, y, 0.20, 0.10)


def test_transitions_detects_synthetic_nesting_violation():
    # P chosen so the real, legitimate transition is two_set -> singleton_1
    # (a genuine shrink, code 3 -> code 2). The size lookup is then
    # corrupted so that singleton_1 compares as larger than two_set,
    # exercising the growth check itself rather than constructing an
    # unreachable violation from valid p-values.
    P = np.array([[0.05, 0.90]])
    y = np.array([1])
    from dslib import alpha_sweep

    original = alpha_sweep._STATE_SIZE
    try:
        alpha_sweep._STATE_SIZE = {0: 0, 1: 1, 2: 3, 3: 2}
        with pytest.raises(ValueError):
            transitions(P, y, 0.01, 0.06)
    finally:
        alpha_sweep._STATE_SIZE = original


# ---------------------------------------------------------------------------
# empty_set_threshold
# ---------------------------------------------------------------------------


def test_empty_set_threshold_matches_reported_qhat():
    est = empty_set_threshold(0.6387313008308411, 0.6331759691238403)
    assert est.sum_q == pytest.approx(1.2719072699546814)
    assert est.empty_possible is False


def test_empty_set_threshold_flags_when_sum_below_one():
    est = empty_set_threshold(0.3, 0.4)
    assert est.sum_q == pytest.approx(0.7)
    assert est.empty_possible is True


# ---------------------------------------------------------------------------
# table_for_paper / to_latex
# ---------------------------------------------------------------------------


def test_table_for_paper_preserves_requested_order():
    P, y = _synthetic_p_values(n=1000)
    alphas = np.round(np.arange(0.01, 0.31, 0.01), 4)
    df = sweep(P, y, alphas)
    tab = table_for_paper(df, [0.20, 0.05, 0.10])
    assert list(tab["alpha"]) == [0.20, 0.05, 0.10]
    assert "TP" in tab.columns and "coverage_class_1" in tab.columns


def test_to_latex_returns_tabular_string():
    P, y = _synthetic_p_values(n=200)
    df = sweep(P, y, [0.10])
    tab = table_for_paper(df, [0.10])
    tex = to_latex(tab)
    assert isinstance(tex, str)
    assert "\\begin{tabular}" in tex
