"""Value function and sampling design for SHAP on conformal p-values.

`experiments/globafa2026/05_shap.py` builds the value function passed to
``shap.KernelExplainer`` as a closure inside a marimo cell, together with the
background/explain sampling that determines which 1000 test instances get
explained. Neither is importable from that notebook. This module factors out
the same logic — unchanged — so it can be reused by a second estimator
(``exact_shapley``) without duplicating or rewriting it, and the calling
notebook is required to assert equivalence against the original closure
before relying on it (see the "Celda de equivalencia" in
``07_exact_shapley.py``).

Categorical features are integer-encoded into a single column each
(`05_shap.py:137-142`); the one-hot expansion happens inside the sklearn
pipeline, after decoding. This keeps the coalition space at ``M = 8``
features (4 numeric + 4 categorical) rather than one per one-hot dummy.

References
----------
Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from conformalpy.classifier import ConformalClassifier


class IntegerEncodedPValueFunction:
    """Value function for KernelSHAP / exact Shapley on conformal p-values.

    Equivalent to the ``predict_p_values_fn`` closure in
    ``experiments/globafa2026/05_shap.py`` (and ``copa2026/05_shap.py``):
    decodes integer-coded categorical columns back to their original labels
    and calls ``conf_clf.predict_p_values`` on the reconstructed DataFrame.
    ``conformalpy.explainability.ConformalPValueFunction`` does not perform
    this decode step, so it cannot be substituted here — passing
    integer-coded categoricals straight to the pipeline's
    ``OneHotEncoder(handle_unknown="ignore")`` would silently zero out every
    categorical dummy.

    Parameters
    ----------
    conf_clf : ConformalClassifier
        Calibrated conformal classifier.
    numeric_cols : list[str]
        Numeric feature names, in the column order of the input array.
    categorical_cols : list[str]
        Categorical feature names, in the column order of the input array
        (following ``numeric_cols``).
    cat_categories : dict[str, list]
        Sorted unique category values per categorical column, as used to
        build the integer encoding (``sorted(X[col].dropna().unique())``).
    """

    def __init__(
        self,
        conf_clf: "ConformalClassifier",
        numeric_cols: list[str],
        categorical_cols: list[str],
        cat_categories: dict[str, list],
    ) -> None:
        self.conf_clf = conf_clf
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cat_categories = cat_categories
        self.all_cols = numeric_cols + categorical_cols
        self.n_num = len(numeric_cols)
        self.int_to_cat = {
            col: dict(enumerate(cats)) for col, cats in cat_categories.items()
        }

    def __call__(self, X: np.ndarray) -> np.ndarray:
        data = {col: X[:, i].astype(float) for i, col in enumerate(self.numeric_cols)}
        for j, col in enumerate(self.categorical_cols):
            n_cats = len(self.cat_categories[col])
            codes = np.round(X[:, self.n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [self.int_to_cat[col][c] for c in codes]
        return self.conf_clf.predict_p_values(pd.DataFrame(data, columns=self.all_cols))


@dataclass
class ShapDesign:
    """Sampling design for SHAP: background/explain split and encoders."""

    X_background: np.ndarray
    X_explain: np.ndarray
    X_explain_df: pd.DataFrame
    background_idx: np.ndarray
    explain_idx: np.ndarray
    y_explain: np.ndarray
    all_cols: list[str]
    cat_categories: dict[str, list]


def build_shap_design(
    X_test: pd.DataFrame,
    y_test,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    seed: int,
    n_background: int,
    n_explain: int,
) -> ShapDesign:
    """Reproduce the background/explain sampling of ``05_shap.py`` exactly.

    Integer-encodes categorical columns, seeds ``np.random`` once, draws
    ``background_idx`` then ``explain_idx`` without replacement from the
    remaining indices — in that order, matching
    ``experiments/globafa2026/05_shap.py:117-160``. Any deviation in the
    order of these two draws changes which 1000 instances get explained.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test-set features, columns including ``numeric_cols + categorical_cols``.
    y_test : array-like
        Test-set labels, aligned with ``X_test``.
    numeric_cols, categorical_cols : list[str]
        Feature names, in the order used to build the encoded matrix.
    seed : int
        Passed to ``np.random.seed`` before sampling.
    n_background, n_explain : int
        Sample sizes for the background and explain sets.

    Returns
    -------
    ShapDesign
    """
    all_cols = numeric_cols + categorical_cols
    cat_categories = {col: sorted(X_test[col].dropna().unique()) for col in categorical_cols}
    cat_to_int = {col: {v: i for i, v in enumerate(cats)} for col, cats in cat_categories.items()}

    X_full = np.column_stack(
        [
            X_test[numeric_cols].values.astype(float),
            np.column_stack([X_test[col].map(cat_to_int[col]).values for col in categorical_cols]),
        ]
    )

    np.random.seed(seed)
    n_test = len(X_test)
    background_idx = np.random.choice(n_test, size=n_background, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(n_test), background_idx), size=n_explain, replace=False
    )

    X_background = X_full[background_idx]
    X_explain = X_full[explain_idx]
    X_explain_df = pd.DataFrame(X_explain, columns=all_cols)
    y_explain = np.asarray(y_test)[explain_idx].astype(int)

    return ShapDesign(
        X_background=X_background,
        X_explain=X_explain,
        X_explain_df=X_explain_df,
        background_idx=background_idx,
        explain_idx=explain_idx,
        y_explain=y_explain,
        all_cols=all_cols,
        cat_categories=cat_categories,
    )


def _all_coalitions(M: int) -> list[tuple[int, ...]]:
    coalitions: list[tuple[int, ...]] = []
    for r in range(M + 1):
        coalitions.extend(itertools.combinations(range(M), r))
    return coalitions


def exact_shapley(f, X_background: np.ndarray, X_explain: np.ndarray) -> np.ndarray:
    """Exact Shapley values by full coalition enumeration, interventional background.

    Computes

        v(S) = mean over the background set of f(x_S, X_background_{-S})

    for every one of the ``2**M`` coalitions ``S``, then the closed-form
    Shapley value

        phi_j(x) = sum_{S subset of N\\{j}} w(|S|) * (v(S union {j}) - v(S))
        w(s) = s! (M - s - 1)! / M!

    Same masking convention (interventional, full background average) as
    ``shap.KernelExplainer`` with the given background set — the two differ
    only in how the ``2**M - 2`` non-trivial coalitions are weighted and
    sampled, not in the value function or the background.

    Parameters
    ----------
    f : callable
        Value function, ``(n, M) -> (n, n_outputs)``.
    X_background : np.ndarray of shape (n_bg, M)
        Background dataset.
    X_explain : np.ndarray of shape (n_ex, M)
        Instances to explain.

    Returns
    -------
    np.ndarray of shape (n_ex, M, n_outputs)

    References
    ----------
    Shapley (1953). A Value for n-Person Games.
    """
    X_background = np.asarray(X_background, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    n_bg, M = X_background.shape
    n_ex = X_explain.shape[0]

    coalitions = _all_coalitions(M)
    coal_index = {S: i for i, S in enumerate(coalitions)}
    n_coal = len(coalitions)

    mask = np.zeros((n_coal, M), dtype=bool)
    for i, S in enumerate(coalitions):
        if S:
            mask[i, list(S)] = True
    mask_repeated = np.repeat(mask, n_bg, axis=0)  # (n_coal * n_bg, M)
    bg_tiled = np.tile(X_background, (n_coal, 1))  # (n_coal * n_bg, M)

    # Per-feature (idx_S, idx_S_with_j, weight) arrays, independent of x.
    idx_S_by_feature = []
    idx_Swj_by_feature = []
    weight_by_feature = []
    for j in range(M):
        others = [k for k in range(M) if k != j]
        idx_S, idx_Swj, weights = [], [], []
        for s in range(M):
            w = factorial(s) * factorial(M - s - 1) / factorial(M)
            for S in itertools.combinations(others, s):
                S_with_j = tuple(sorted(S + (j,)))
                idx_S.append(coal_index[S])
                idx_Swj.append(coal_index[S_with_j])
                weights.append(w)
        idx_S_by_feature.append(np.array(idx_S))
        idx_Swj_by_feature.append(np.array(idx_Swj))
        weight_by_feature.append(np.array(weights))

    phi = None
    for i in range(n_ex):
        x = X_explain[i]
        design = np.where(mask_repeated, x[None, :], bg_tiled)
        preds = np.asarray(f(design))
        n_outputs = preds.shape[-1] if preds.ndim > 1 else 1
        preds = preds.reshape(n_coal, n_bg, n_outputs)
        v = preds.mean(axis=1)  # (n_coal, n_outputs)

        if phi is None:
            phi = np.zeros((n_ex, M, n_outputs))

        for j in range(M):
            diffs = v[idx_Swj_by_feature[j]] - v[idx_S_by_feature[j]]
            phi[i, j] = weight_by_feature[j] @ diffs

    return phi
