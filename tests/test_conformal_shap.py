"""Tests for dslib.conformal_shap — value function, sampling design, exact Shapley."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dslib.conformal_shap import (
    IntegerEncodedPValueFunction,
    build_shap_design,
    exact_shapley,
)


# ---------------------------------------------------------------------------
# build_shap_design
# ---------------------------------------------------------------------------


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "cat2": rng.choice(["x", "y"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


def test_build_shap_design_shapes_and_no_overlap():
    X, y = _toy_frame()
    design = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=42, n_background=30, n_explain=50)
    assert design.X_background.shape == (30, 4)
    assert design.X_explain.shape == (50, 4)
    assert design.y_explain.shape == (50,)
    assert len(set(design.background_idx.tolist()) & set(design.explain_idx.tolist())) == 0


def test_build_shap_design_is_deterministic_given_seed():
    X, y = _toy_frame()
    d1 = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=42, n_background=30, n_explain=50)
    d2 = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=42, n_background=30, n_explain=50)
    assert np.array_equal(d1.background_idx, d2.background_idx)
    assert np.array_equal(d1.explain_idx, d2.explain_idx)


def test_build_shap_design_different_seed_changes_indices():
    X, y = _toy_frame()
    d1 = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=1, n_background=30, n_explain=50)
    d2 = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=2, n_background=30, n_explain=50)
    assert not np.array_equal(d1.explain_idx, d2.explain_idx)


# ---------------------------------------------------------------------------
# IntegerEncodedPValueFunction
# ---------------------------------------------------------------------------


class _FakeConfClf:
    """Records the DataFrame it was called with and returns fixed p-values."""

    def __init__(self):
        self.last_df = None

    def predict_p_values(self, df):
        self.last_df = df
        return np.column_stack([np.full(len(df), 0.4), np.full(len(df), 0.6)])


def test_integer_encoded_pvalue_function_decodes_categoricals():
    X, y = _toy_frame()
    design = build_shap_design(X, y, ["num1", "num2"], ["cat1", "cat2"], seed=42, n_background=30, n_explain=50)

    clf = _FakeConfClf()
    f = IntegerEncodedPValueFunction(clf, ["num1", "num2"], ["cat1", "cat2"], design.cat_categories)
    out = f(design.X_explain)

    assert out.shape == (50, 2)
    assert clf.last_df["cat1"].isin(["a", "b", "c"]).all()
    assert clf.last_df["cat2"].isin(["x", "y"]).all()
    # numeric columns pass through unchanged (up to dtype)
    np.testing.assert_allclose(clf.last_df["num1"].values, design.X_explain[:, 0])


def test_integer_encoded_pvalue_function_matches_reference_closure():
    # The reference closure from experiments/globafa2026/05_shap.py, reproduced
    # verbatim, to confirm IntegerEncodedPValueFunction is a drop-in replacement.
    X, y = _toy_frame(n=300, seed=3)
    numeric_cols, categorical_cols = ["num1", "num2"], ["cat1", "cat2"]
    all_cols = numeric_cols + categorical_cols
    n_num = len(numeric_cols)

    cat_categories = {col: sorted(X[col].dropna().unique()) for col in categorical_cols}
    cat_to_int = {col: {v: i for i, v in enumerate(cats)} for col, cats in cat_categories.items()}
    int_to_cat = {col: {i: v for v, i in m.items()} for col, m in cat_to_int.items()}

    X_full = np.column_stack(
        [
            X[numeric_cols].values.astype(float),
            np.column_stack([X[col].map(cat_to_int[col]).values for col in categorical_cols]),
        ]
    )

    clf = _FakeConfClf()

    def reference_closure(Z: np.ndarray) -> np.ndarray:
        data = {col: Z[:, i].astype(float) for i, col in enumerate(numeric_cols)}
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(Z[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        return clf.predict_p_values(pd.DataFrame(data, columns=all_cols))

    f = IntegerEncodedPValueFunction(clf, numeric_cols, categorical_cols, cat_categories)

    out_ref = reference_closure(X_full[:20])
    out_new = f(X_full[:20])
    assert np.array_equal(out_ref, out_new)


# ---------------------------------------------------------------------------
# exact_shapley
# ---------------------------------------------------------------------------


def _nonadditive_value_fn(X: np.ndarray) -> np.ndarray:
    """Non-additive, discontinuous, vector-valued function — a stand-in for
    a conformal p-value's step-function behaviour."""
    X = np.atleast_2d(X)
    a = np.floor(3 * X[:, 0]) * X[:, 1] + np.abs(X[:, 2] - X[:, 3])
    if X.shape[1] > 4:
        a = a + (X[:, 4] > 0.5) * X[:, 5] if X.shape[1] > 5 else a + X[:, 4]
    a = np.round(a, 2)
    return np.column_stack([1 - a / 5, a / 5])


def test_exact_shapley_matches_kernelshap_full_enumeration():
    shap = pytest.importorskip("shap")
    rng = np.random.default_rng(7)
    M = 6
    bg = rng.random((40, M))
    x = rng.random((4, M))

    phi = exact_shapley(_nonadditive_value_fn, bg, x)
    assert phi.shape == (4, M, 2)

    explainer = shap.KernelExplainer(_nonadditive_value_fn, bg)
    sv = np.array(explainer.shap_values(x, nsamples=2**M - 2))

    np.testing.assert_allclose(phi, sv, atol=1e-9)


def test_exact_shapley_satisfies_efficiency_property():
    # sum_j phi_j(x) == f(x) - f(background_mean), the additivity/efficiency
    # axiom of Shapley values, independent of any reference implementation.
    rng = np.random.default_rng(11)
    M = 5
    bg = rng.random((30, M))
    x = rng.random((2, M))

    phi = exact_shapley(_nonadditive_value_fn, bg, x)
    fx = _nonadditive_value_fn(x)
    f_bg_mean = _nonadditive_value_fn(bg).mean(axis=0)

    total = phi.sum(axis=1)  # (n_ex, n_outputs)
    expected = fx - f_bg_mean[None, :]
    np.testing.assert_allclose(total, expected, atol=1e-9)


def test_exact_shapley_single_feature_gets_full_effect():
    def f(X):
        X = np.atleast_2d(X)
        return (2 * X[:, 0]).reshape(-1, 1)

    bg = np.zeros((5, 1))
    x = np.array([[3.0]])
    phi = exact_shapley(f, bg, x)
    assert phi.shape == (1, 1, 1)
    np.testing.assert_allclose(phi[0, 0], [6.0])
