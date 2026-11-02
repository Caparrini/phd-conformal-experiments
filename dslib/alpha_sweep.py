"""Alpha sensitivity analysis for a calibrated Mondrian LAC conformal classifier.

Under the LAC nonconformity function, ``s(x, y) = 1 - p_hat(y | x)``
(``conformalpy.nonconformity.classification.lac_nonconformity``), the conservative
conformal p-value returned by ``ConformalClassifier.predict_p_values`` is exactly
equivalent to set membership at every alpha:

    C_alpha(x) = {y : p_y(x) > alpha}

(``ConformalClassifier.predict_p_values`` docstring; the two are related by
``p_y(x) = (N_> + N_= + 1) / (N_cal + 1)`` per class in Mondrian mode). A single
p-value matrix ``P`` of shape ``(n, n_classes)`` therefore determines the
prediction sets for every alpha without recomputing calibration quantiles.

This module takes ``P`` and the true labels ``y`` as its only inputs. It does not
import ``conformalpy``'s classifier, MLflow, or Hydra — the caller (an experiment
notebook) is responsible for producing ``P`` via a calibrated
``ConformalClassifier`` and for logging the results. Binary classification only
(class 0 / class 1), matching the LAC/Mondrian setup used in ``globafa2026``.

References
----------
Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from conformalpy.evaluation.classification_metrics import (
    average_set_size,
    class_conditional_coverage,
    coverage_score,
    empty_sets_score,
    one_class_correct_score,
    one_class_incorrect_score,
    two_class_sets_score,
)
from conformalpy.outcomes.categorization import singleton_confusion_matrix
from conformalpy.plots.classification import plot_coverage_vs_confidence

_SUMMARY_COLUMNS = [
    "Confidence",
    "Coverage",
    "One_Class_Correct",
    "One_Class_Incorrect",
    "Two_Class_Sets",
    "No_Sets",
]


def sets_from_p_values(P: np.ndarray, alpha: float) -> list[set]:
    """Build binary prediction sets ``{y : p_y > alpha}`` from a p-value matrix.

    Parameters
    ----------
    P : np.ndarray of shape (n, 2)
        Conformal p-values, columns ``[p0, p1]``.
    alpha : float
        Miscoverage level.

    Returns
    -------
    list of set
        One prediction set per row of ``P``, drawn from ``{0}``, ``{1}``,
        ``{0, 1}``, ``set()``.
    """
    P = np.asarray(P)
    in0 = P[:, 0] > alpha
    in1 = P[:, 1] > alpha
    sets: list[set] = []
    for a, b in zip(in0, in1):
        if a and b:
            sets.append({0, 1})
        elif a:
            sets.append({0})
        elif b:
            sets.append({1})
        else:
            sets.append(set())
    return sets


def _singleton_class(P: np.ndarray) -> np.ndarray:
    """Class each sample resolves to once it becomes a singleton.

    For a fixed sample, class 0 leaves the set at ``alpha = p0`` and class 1
    leaves at ``alpha = p1``; the surviving class in the singleton band
    ``min(p0, p1) <= alpha < max(p0, p1)`` is the one with the larger p-value.
    This does not depend on which alpha in that band is queried.
    """
    return np.argmax(np.asarray(P), axis=1)


def _band_masks(P: np.ndarray, alphas: np.ndarray) -> list[np.ndarray]:
    """Boolean masks of samples entering singleton status in each alpha band.

    Band ``k`` is ``[alphas[k-1], alphas[k])`` for ``k > 0`` and
    ``[0, alphas[0])`` for ``k = 0``, applied to ``min(p0, p1)`` — the alpha at
    which a sample stops being a two-set.
    """
    min_p = np.minimum(P[:, 0], P[:, 1])
    edges = np.concatenate(([0.0], alphas))
    return [
        (min_p >= edges[k]) & (min_p < edges[k + 1])
        for k in range(len(alphas))
    ]


def sweep(P: np.ndarray, y: np.ndarray, alphas) -> pd.DataFrame:
    """Evaluate a Mondrian LAC conformal predictor across a grid of alpha.

    One p-value matrix suffices for the whole grid (see module docstring);
    no recalibration is performed.

    Parameters
    ----------
    P : np.ndarray of shape (n, 2)
        Conformal p-values, columns ``[p0, p1]``.
    y : array-like of shape (n,)
        True binary labels (0 or 1).
    alphas : array-like of float
        Miscoverage levels to evaluate, in the order given.

    Returns
    -------
    pd.DataFrame
        One row per alpha. Absolute counts ``TP``, ``FP``, ``TN``, ``FN``
        (singleton confusion, class 1 positive), ``TS`` (two-set count),
        ``E`` (empty count); rate columns ``coverage``, ``coverage_class_0``,
        ``coverage_class_1``, ``avg_set_size``, ``singleton_rate``,
        ``two_class_rate``, ``empty_rate``, ``accuracy_given_confident``;
        per-class miscoverage slack against the nominal bound
        ``coverage_class_y - (1 - alpha)``; ``recruit_frac_class1`` (fraction
        of newly-formed singletons in this alpha's band assigned to class 1);
        and the aliases ``Confidence``, ``Coverage``, ``One_Class_Correct``,
        ``One_Class_Incorrect``, ``Two_Class_Sets``, ``No_Sets`` required by
        ``conformalpy.plots.plot_coverage_vs_confidence``.
    """
    P = np.asarray(P)
    y = np.asarray(y)
    alphas = np.asarray(sorted(alphas), dtype=float) if not _is_sorted(alphas) else np.asarray(alphas, dtype=float)

    singleton_class = _singleton_class(P)
    band_masks = _band_masks(P, alphas)

    rows = []
    for alpha, band_mask in zip(alphas, band_masks):
        prediction_sets = sets_from_p_values(P, alpha)
        sizes = np.array([len(s) for s in prediction_sets], dtype=np.int64)

        scm = singleton_confusion_matrix(
            prediction_sets, y, class_names={0: "Not Default", 1: "Default"}
        )
        TN, FP, FN, TP = (
            scm["counts"]["TN"],
            scm["counts"]["FP"],
            scm["counts"]["FN"],
            scm["counts"]["TP"],
        )
        TS = int(np.sum(sizes == 2))
        E = int(np.sum(sizes == 0))

        coverage = coverage_score(prediction_sets, y)
        ccc = class_conditional_coverage(prediction_sets, y, classes=[0, 1])
        avg_size = average_set_size(prediction_sets)
        sc_rate = one_class_correct_score(prediction_sets, y)
        si_rate = one_class_incorrect_score(prediction_sets, y)
        ts_rate = two_class_sets_score(prediction_sets)
        empty_rate = empty_sets_score(prediction_sets)

        n_confident = TP + FP + TN + FN
        accuracy_given_confident = (TP + TN) / n_confident if n_confident > 0 else np.nan

        n_recruited = int(np.sum(band_mask))
        recruit_frac_class1 = (
            float(np.mean(singleton_class[band_mask] == 1)) if n_recruited > 0 else np.nan
        )

        rows.append(
            {
                "alpha": float(alpha),
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "TS": TS,
                "E": E,
                "n_singletons": TP + FP + TN + FN,
                "coverage": coverage,
                "coverage_class_0": ccc.get(0, np.nan),
                "coverage_class_1": ccc.get(1, np.nan),
                "miscoverage_slack_class_0": ccc.get(0, np.nan) - (1 - alpha),
                "miscoverage_slack_class_1": ccc.get(1, np.nan) - (1 - alpha),
                "avg_set_size": avg_size,
                "singleton_rate": sc_rate + si_rate,
                "two_class_rate": ts_rate,
                "empty_rate": empty_rate,
                "accuracy_given_confident": accuracy_given_confident,
                "n_recruited": n_recruited,
                "recruit_frac_class1": recruit_frac_class1,
                "Confidence": 1 - alpha,
                "Coverage": coverage,
                "One_Class_Correct": sc_rate,
                "One_Class_Incorrect": si_rate,
                "Two_Class_Sets": ts_rate,
                "No_Sets": empty_rate,
            }
        )

    return pd.DataFrame(rows)


def _is_sorted(a) -> bool:
    a = list(a)
    return a == sorted(a)


def marginal_precision(P: np.ndarray, y: np.ndarray, alphas) -> pd.DataFrame:
    """Precision of newly-formed singletons within each alpha band.

    Computed directly on the band ``alpha_k <= min(p0, p1) < alpha_{k+1}``,
    not from differences of cumulative singleton accuracy — the latter is
    noisy under a coarse grid because it mixes samples that entered the
    singleton state at different points in the range.

    Parameters
    ----------
    P : np.ndarray of shape (n, 2)
        Conformal p-values, columns ``[p0, p1]``.
    y : array-like of shape (n,)
        True binary labels.
    alphas : array-like of float
        Increasing grid of miscoverage levels; band ``k`` is
        ``[alphas[k-1], alphas[k])`` (``[0, alphas[0])`` for ``k = 0``).

    Returns
    -------
    pd.DataFrame
        One row per band: ``alpha_lo``, ``alpha_hi``, ``n_entrants``,
        ``n_correct``, ``precision``, ``class1_frac``.
    """
    P = np.asarray(P)
    y = np.asarray(y)
    alphas = np.asarray(alphas, dtype=float)
    if not _is_sorted(alphas):
        raise ValueError("alphas must be sorted ascending")

    singleton_class = _singleton_class(P)
    band_masks = _band_masks(P, alphas)
    edges = np.concatenate(([0.0], alphas))

    rows = []
    for k, mask in enumerate(band_masks):
        n_entrants = int(np.sum(mask))
        if n_entrants == 0:
            rows.append(
                {
                    "alpha_lo": edges[k],
                    "alpha_hi": edges[k + 1],
                    "n_entrants": 0,
                    "n_correct": 0,
                    "precision": np.nan,
                    "class1_frac": np.nan,
                }
            )
            continue
        pred = singleton_class[mask]
        correct = pred == y[mask]
        rows.append(
            {
                "alpha_lo": edges[k],
                "alpha_hi": edges[k + 1],
                "n_entrants": n_entrants,
                "n_correct": int(np.sum(correct)),
                "precision": float(np.mean(correct)),
                "class1_frac": float(np.mean(pred == 1)),
            }
        )
    return pd.DataFrame(rows)


_STATE_NAMES = {0: "empty", 1: "singleton_0", 2: "singleton_1", 3: "two_set"}


def _state_labels(P: np.ndarray, alpha: float) -> np.ndarray:
    """Encode each sample's set state at a given alpha as an integer code.

    Codes follow set size (0 = empty, 2 = two-set); singletons are split by
    class since a migration between classes at fixed size is a distinct event.
    """
    in0 = P[:, 0] > alpha
    in1 = P[:, 1] > alpha
    code = np.zeros(len(P), dtype=np.int8)
    code[in0 & ~in1] = 1
    code[~in0 & in1] = 2
    code[in0 & in1] = 3
    return code


_STATE_SIZE = {0: 0, 1: 1, 2: 1, 3: 2}


def transitions(P: np.ndarray, y: np.ndarray, a1: float, a2: float) -> pd.DataFrame:
    """Migration matrix of set states between two alpha levels.

    Parameters
    ----------
    P : np.ndarray of shape (n, 2)
        Conformal p-values, columns ``[p0, p1]``.
    y : array-like of shape (n,)
        True binary labels (unused in the matrix; accepted for symmetry with
        the rest of the module's API and for future class-conditional
        breakdowns).
    a1, a2 : float
        Alpha levels compared, with ``a1 < a2``.

    Returns
    -------
    pd.DataFrame
        4x4 count matrix indexed by state at ``a1`` (rows) and state at
        ``a2`` (columns), states in
        ``{"empty", "singleton_0", "singleton_1", "two_set"}``.

    Raises
    ------
    ValueError
        If ``a1 >= a2``, or if any sample's set grows from ``a1`` to ``a2``
        (nesting violation — prediction sets must shrink or stay the same
        size as alpha increases).
    """
    if a1 >= a2:
        raise ValueError(f"a1 must be < a2, got a1={a1}, a2={a2}")

    s1 = _state_labels(P, a1)
    s2 = _state_labels(P, a2)

    size1 = np.vectorize(_STATE_SIZE.get)(s1)
    size2 = np.vectorize(_STATE_SIZE.get)(s2)
    grew = size2 > size1
    if np.any(grew):
        raise ValueError(
            f"Nesting violated: {int(np.sum(grew))} sample(s) grew from alpha={a1} "
            f"to alpha={a2}; prediction sets must be non-increasing in alpha."
        )

    labels = [_STATE_NAMES[k] for k in range(4)]
    mat = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            mat[i, j] = int(np.sum((s1 == i) & (s2 == j)))

    return pd.DataFrame(mat, index=labels, columns=labels)


@dataclass
class EmptySetThreshold:
    """Analytical check for whether empty prediction sets are possible.

    Under binary LAC, ``s(x, 0) + s(x, 1) = (1 - p_hat_0(x)) + (1 - p_hat_1(x)) = 1``
    for any x (the two class probabilities sum to 1), so the calibration
    quantiles ``q0``, ``q1`` alone determine whether an empty set can occur:
    a sample is empty only if both nonconformity scores exceed their
    respective threshold, which requires ``q0 + q1 < 1``.
    """

    q0: float
    q1: float

    @property
    def sum_q(self) -> float:
        return self.q0 + self.q1

    @property
    def empty_possible(self) -> bool:
        return self.sum_q < 1.0


def empty_set_threshold(q0: float, q1: float) -> EmptySetThreshold:
    """Check analytically whether ``q0 + q1`` rules out empty prediction sets.

    Parameters
    ----------
    q0, q1 : float
        Mondrian LAC calibration quantiles for class 0 and class 1.

    Returns
    -------
    EmptySetThreshold
        ``sum_q = q0 + q1`` and ``empty_possible = sum_q < 1``.
    """
    return EmptySetThreshold(q0=q0, q1=q1)


def check(df: pd.DataFrame, P: np.ndarray, y: np.ndarray) -> None:
    """Verify identities that must hold for any correct alpha sweep.

    Raises ``AssertionError`` with the offending row and quantity on the
    first violation found.

    Checks
    ------
    1. ``TP + FP + TN + FN + TS + E == n`` for every row.
    2. ``coverage * n`` matches the count of samples covered by construction:
       singleton-correct (``TP + TN``) plus every two-set (a two-set always
       contains both classes in binary classification).
    3. ``avg_set_size`` is non-increasing as alpha increases.
    4. ``singleton_rate + two_class_rate + empty_rate == 1`` for every row
       (set sizes are drawn from {0, 1, 2} only in binary classification).
    """
    n = len(y)
    df_sorted = df.sort_values("alpha").reset_index(drop=True)

    totals = df_sorted["TP"] + df_sorted["FP"] + df_sorted["TN"] + df_sorted["FN"] + df_sorted["TS"] + df_sorted["E"]
    bad = df_sorted.index[totals != n]
    if len(bad) > 0:
        i = bad[0]
        raise AssertionError(
            f"Row alpha={df_sorted.loc[i, 'alpha']}: outcome counts sum to "
            f"{totals[i]}, expected n={n}."
        )

    covered_count = df_sorted["TP"] + df_sorted["TN"] + df_sorted["TS"]
    implied_coverage = covered_count / n
    mismatch = (df_sorted["coverage"] - implied_coverage).abs() > 1e-9
    bad = df_sorted.index[mismatch]
    if len(bad) > 0:
        i = bad[0]
        raise AssertionError(
            f"Row alpha={df_sorted.loc[i, 'alpha']}: coverage={df_sorted.loc[i, 'coverage']} "
            f"does not match outcome-implied coverage={implied_coverage[i]}."
        )

    size_diff = df_sorted["avg_set_size"].diff()
    bad = df_sorted.index[(size_diff > 1e-9) & df_sorted.index.to_series().gt(0)]
    if len(bad) > 0:
        i = bad[0]
        raise AssertionError(
            f"avg_set_size increased at alpha={df_sorted.loc[i, 'alpha']} "
            f"({df_sorted.loc[i - 1, 'avg_set_size']} -> {df_sorted.loc[i, 'avg_set_size']}); "
            "prediction sets must shrink or stay the same size as alpha increases."
        )

    rate_sum = df_sorted["singleton_rate"] + df_sorted["two_class_rate"] + df_sorted["empty_rate"]
    bad = df_sorted.index[(rate_sum - 1.0).abs() > 1e-9]
    if len(bad) > 0:
        i = bad[0]
        raise AssertionError(
            f"Row alpha={df_sorted.loc[i, 'alpha']}: singleton_rate + two_class_rate + "
            f"empty_rate = {rate_sum[i]}, expected 1."
        )


def plot_evolution(df: pd.DataFrame, base_rate: float | None = None):
    """Evolution of coverage and set composition across the alpha grid.

    Thin wrapper over ``conformalpy.plots.plot_coverage_vs_confidence`` (the
    stacked-area coverage/composition plot), adding TP/FP/TN/FN rate lines
    on a secondary axis and, if given, a horizontal reference at
    ``base_rate`` — the accuracy achievable by always predicting the
    majority class, below which raising alpha stops being informative.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``sweep``.
    base_rate : float, optional
        Majority-class base rate, e.g. ``1 - y.mean()``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("Confidence")
    fig, ax = plot_coverage_vs_confidence(df_sorted[_SUMMARY_COLUMNS])

    n = df_sorted["TP"] + df_sorted["FP"] + df_sorted["TN"] + df_sorted["FN"]
    ax2 = ax.twinx()
    for col, color in [("TP", "#1f77b4"), ("FP", "#d62728"), ("TN", "#2ca02c"), ("FN", "#9467bd")]:
        rate = df_sorted[col] / n.replace(0, np.nan)
        ax2.plot(df_sorted["Confidence"], rate, linestyle=":", color=color, label=f"{col} rate")
    ax2.set_ylabel("Singleton outcome rate")
    ax2.legend(loc="center right", fontsize="small")

    if base_rate is not None:
        ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1, label="Base rate")

    plt.tight_layout()
    return fig, ax


def plot_risk_coverage(df: pd.DataFrame):
    """Risk-coverage curve: error rate among confident predictions vs.
    the fraction of predictions made with confidence (singleton rate).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``sweep``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    References
    ----------
    El-Yaniv & Wiener (2010). On the Foundations of Noise-free Selective
    Classification.
    """
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("singleton_rate")
    risk = 1 - df_sorted["accuracy_given_confident"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df_sorted["singleton_rate"], risk, "o-")
    for _, row in df_sorted.iterrows():
        ax.annotate(f"α={row['alpha']:.2f}", (row["singleton_rate"], 1 - row["accuracy_given_confident"]),
                     fontsize="small", textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Coverage (singleton rate)")
    ax.set_ylabel("Risk (1 - accuracy | singleton)")
    ax.set_title("Risk-coverage curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def table_for_paper(df: pd.DataFrame, alphas) -> pd.DataFrame:
    """Select and order the columns cited in the paper for a subset of alphas.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``sweep``.
    alphas : array-like of float
        Alpha values to include, in the given order.

    Returns
    -------
    pd.DataFrame
        Columns: alpha, coverage, coverage_class_0, coverage_class_1,
        avg_set_size, singleton_rate, two_class_rate, empty_rate,
        accuracy_given_confident, TP, FP, TN, FN.
    """
    cols = [
        "alpha",
        "coverage",
        "coverage_class_0",
        "coverage_class_1",
        "avg_set_size",
        "singleton_rate",
        "two_class_rate",
        "empty_rate",
        "accuracy_given_confident",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    indexed = df.set_index("alpha")
    rows = [indexed.loc[a] for a in alphas]
    out = pd.DataFrame(rows)
    out.index.name = "alpha"
    out = out.reset_index()
    return out[cols]


def to_latex(tab: pd.DataFrame) -> str:
    """Render a table (typically from ``table_for_paper``) as a LaTeX tabular."""
    return tab.to_latex(index=False, float_format="%.4f")


if __name__ == "__main__":
    # Synthetic validation of the module's mechanics. p0 is constructed
    # coarsely on purpose (rounded to 2 decimals) to exercise the discrete
    # ties that occur near alpha -> 0 with a finite calibration set; these
    # numbers are not representative of the shape expected from real
    # conformal p-values and should not be used to anticipate the paper's
    # figures.
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, size=n)
    base = rng.beta(2, 2, size=n)
    p1 = np.where(y == 1, 1 - base, base)
    p0 = np.round(1 - p1 + rng.normal(0, 0.05, size=n), 2)
    p0 = np.clip(p0, 0.0, 1.0)
    P = np.column_stack([p0, p1])

    fine = np.round(np.arange(0.01, 0.31, 0.01), 4)
    df = sweep(P, y, fine)
    check(df, P, y)

    tab = table_for_paper(df, [0.05, 0.10, 0.15, 0.20])
    print(tab.to_string(index=False))

    mp = marginal_precision(P, y, fine)
    print(mp.to_string(index=False))

    tr = transitions(P, y, 0.05, 0.10)
    print(tr)

    est = empty_set_threshold(0.6387313008308411, 0.6331759691238403)
    print(f"q0 + q1 = {est.sum_q:.6f}, empty_possible = {est.empty_possible}")
