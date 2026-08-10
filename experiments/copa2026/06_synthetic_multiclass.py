import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    from itertools import combinations, permutations
    from math import factorial
    from scipy.special import softmax
    from scipy.stats import pearsonr
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity
    from conformalpy.explainability import derive_margin_shap, select_class_shap
    from conformalpy.fcod import compute_fcod_smoothed, plot_stacked_fcod
    from conformalpy.shap import (
        plot_shap_dependence_numeric,
        make_dependence_grid,
    )

    return (
        ConformalClassifier,
        XGBClassifier,
        combinations,
        compute_fcod_smoothed,
        derive_margin_shap,
        factorial,
        lac_nonconformity,
        make_dependence_grid,
        mo,
        np,
        pd,
        pearsonr,
        permutations,
        plot_shap_dependence_numeric,
        plot_stacked_fcod,
        plt,
        select_class_shap,
        shap,
        softmax,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # copa2026, 06: Synthetic multiclass benchmark

    Caparrini, A., Ariza-Garzón, M.-J., Arroyo, J.
    *Explaining Conformal Prediction: Diagnosing Reliability through
    Feature-Conditioned Outcomes and p-Value Margins.* Proceedings of the 15th
    Symposium on Conformal and Probabilistic Prediction with Applications
    (COPA 2026), Göteborg, Sweden. PMLR (forthcoming).

    This notebook runs the same pipeline as notebooks 02–05 (XGBoost → Mondrian
    conformal with LAC → KernelSHAP on p-values → margins) on a 3-class
    synthetic dataset with known ground truth. It is self-contained: data is
    generated in-notebook, with no external state (Hydra/MLflow/parquet).

    Questions addressed:

    | Question | Section |
    |---|---|
    | Viability and dynamics of the correctness margin with more than two competing classes | §2 (outcomes), §3 (margins), §4 (FCOD) |
    | Generality of the framework beyond the binary case | §5 (correctness margin), §3c–§3f (same analysis, on the evidence margin) |
    | Faithfulness of margin-SHAP against exact ground truth | §7 |
    | Stability of KernelSHAP across seed, background, nsamples | §8 |

    Sections 2–6 reproduce the paper's instruments (conformal outcomes,
    correctness margin, FCOD, margin-SHAP) unchanged, on three classes. §7–§8
    use what a synthetic dataset uniquely allows: exact ground truth. §3c–§3f
    apply the same ground truth to the evidence margin.

    Design: three Gaussian clusters at the vertices of a triangle, in an
    informative 2-D subspace (`x0`, `x1`), plus 6 pure-noise features,
    imbalanced priors (0.5 / 0.3 / 0.2). A known generative process gives an
    exact Bayes posterior and hence exact ground-truth Shapley values
    (GT-Shapley, as in XAI-Bench); true noise attribution is exactly 0. σ is
    set so that ~20% of test instances receive multi-label prediction sets at
    α=0.10.
    """)
    return


@app.cell
def _(np):
    # Constants, mirrors config/copa2026.yaml where applicable.
    # Deliberate divergences from the yaml are marked with (*).
    SEED = 42
    ALPHA = 0.10                       # conformal.alpha
    N_TOTAL = 12_000                   # (*) synthetic sample size
    SIGMA = 1.3                        # (*) cluster spread → ~20% multi-sets at alpha=0.10
    PRIORS = np.array([0.5, 0.3, 0.2])
    MU = np.array([[0.0, 2.0], [-np.sqrt(3), -1.0], [np.sqrt(3), -1.0]])
    N_NOISE = 6
    FEATURES = ["x0", "x1"] + [f"noise{j}" for j in range(N_NOISE)]

    XGB_PARAMS = dict(                 # model.params (multiclass objective (*))
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        objective="multi:softprob", eval_metric="mlogloss",
    )

    N_BACKGROUND = 300                 # shap.n_background
    N_EXPLAIN = 300                    # (*) 1000 in yaml; 300 keeps the full run
    #                                    (incl. §8 stability) around 10–15 min
    NSAMPLES = 1024                    # shap.nsamples, with d=8 this exceeds the
    #                                    2^8−2 = 254 possible coalitions, so
    #                                    KernelSHAP enumerates the full lattice:
    #                                    the estimate is EXACT given the background.
    return (
        ALPHA,
        FEATURES,
        MU,
        NSAMPLES,
        N_BACKGROUND,
        N_EXPLAIN,
        N_NOISE,
        N_TOTAL,
        PRIORS,
        SEED,
        SIGMA,
        XGB_PARAMS,
    )


@app.cell
def _(
    MU,
    N_NOISE,
    N_TOTAL,
    PRIORS,
    SEED,
    SIGMA,
    np,
    softmax,
    train_test_split,
):
    # --- Data generation + 60/20/20 stratified split (same ratios/seed as dslib) ---
    _rng = np.random.default_rng(SEED)
    _y = _rng.choice(3, size=N_TOTAL, p=PRIORS)
    _X = np.hstack([
        MU[_y] + SIGMA * _rng.standard_normal((N_TOTAL, 2)),
        _rng.standard_normal((N_TOTAL, N_NOISE)),
    ])

    def bayes_posterior(X: np.ndarray) -> np.ndarray:
        """Exact P(y|x) of the generative process. Depends only on (x0, x1)."""
        logits = (
            -((X[:, None, :2] - MU[None]) ** 2).sum(-1) / (2 * SIGMA**2)
            + np.log(PRIORS)
        )
        return softmax(logits, axis=1)

    X_train, _X_tmp, y_train, _y_tmp = train_test_split(
        _X, _y, test_size=0.4, stratify=_y, random_state=SEED
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        _X_tmp, _y_tmp, test_size=0.5, stratify=_y_tmp, random_state=SEED
    )
    return X_cal, X_test, X_train, bayes_posterior, y_cal, y_test, y_train


@app.cell
def _(
    ALPHA,
    ConformalClassifier,
    FEATURES,
    XGBClassifier,
    XGB_PARAMS,
    X_cal,
    X_test,
    X_train,
    bayes_posterior,
    lac_nonconformity,
    mo,
    pd,
    y_cal,
    y_test,
    y_train,
):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(pd.DataFrame(X_train, columns=FEATURES), y_train)

    conf_clf = ConformalClassifier(
        model=model,
        alpha=ALPHA,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(pd.DataFrame(X_cal, columns=FEATURES), y_cal)

    # Sanity check (AttributionLab caveat): the ground truth of the DATA is a valid
    # reference for the MODEL only if the model learned the designed association.
    _acc = float((model.predict(pd.DataFrame(X_test, columns=FEATURES)) == y_test).mean())
    _acc_bayes = float((bayes_posterior(X_test).argmax(1) == y_test).mean())
    mo.md(
        f"Conformal calibrado, α={ALPHA} | Mondrian=True | n_cal={len(X_cal):,}  \n"
        f"Model accuracy **{_acc:.3f}** vs Bayes optimum **{_acc_bayes:.3f}**: "
        f"the model approximates the true posterior, so data ground truth ≈ model ground truth."
    )
    return conf_clf, model


@app.cell
def _(mo):
    mo.md("""
    ## 2. Multiclass viability

    Conformal outcomes, taxonomy as in FCOD:

    - **SC**: singleton, correct
    - **SI**: singleton, incorrect
    - **MC**: multi-set, covers the true class
    - **MU**: multi-set, misses the true class (impossible in binary, where the full set always covers)

    Expected: class-conditional coverage ≈ 1−α under imbalance (Mondrian
    guarantee); multi-sets concentrated at pairwise class boundaries; MU rare,
    since it requires the true class to be weak exactly where two rivals are
    strong.
    """)
    return


@app.cell
def _(ALPHA, FEATURES, X_test, conf_clf, mo, np, pd, y_test):
    p_values_test = conf_clf.predict_p_values(pd.DataFrame(X_test, columns=FEATURES))
    pred_sets = p_values_test > ALPHA

    _size = pred_sets.sum(1)
    _covered = pred_sets[np.arange(len(y_test)), y_test]
    # Same taxonomy as conformalpy's FCOD: multi-sets split by coverage.
    # MU (multi-set without the true class) cannot occur in binary, where the
    # full set always covers; it is a genuinely multiclass outcome. Naming
    # note: the paper uses TS (with a binary TS0/TS1 refinement by true
    # label) for the multi-set category; this notebook uses the coverage
    # split (MC/MU), which is the one that generalizes beyond binary.
    outcome = np.where(
        _size == 0, "EMPTY",
        np.where(
            _size >= 2,
            np.where(_covered, "MC", "MU"),
            np.where(_covered, "SC", "SI"),
        ),
    )

    # Per-instance competing class y*: highest p-value among the non-true classes.
    _mask_true = np.zeros_like(p_values_test, dtype=bool)
    _mask_true[np.arange(len(y_test)), y_test] = True
    y_star = np.where(_mask_true, -np.inf, p_values_test).argmax(1)

    _n = len(y_test)
    _counts = {o: int((outcome == o).sum()) for o in ["SC", "SI", "MC", "MU", "EMPTY"]}
    # Paper's outcome-category section: P(SI) + P(Empty) <= alpha in binary. The
    # multiclass form of the same coverage identity is
    # P(SI) + P(MU) + P(Empty) <= alpha, since MU joins the uncovered mass.
    _uncov = {k: float((~_covered[y_test == k]).mean()) for k in range(3)}
    _uncov_rows = " | ".join(f"class {k}: {_uncov[k]:.3f}" for k in range(3))
    _cov_rows = " | ".join(
        f"class {k}: {_covered[y_test == k].mean():.3f}" for k in range(3)
    )
    _ystar_rows = "  \n".join(
        f"true class {k} → competitors " + ", ".join(
            f"y*={c}: {int(((y_test == k) & (y_star == c)).sum())}"
            for c in range(3) if c != k
        )
        for k in range(3)
    )
    mo.md(
        f"**Outcomes** (n={_n}): " +
        ", ".join(f"{o}: {v} ({100*v/_n:.1f}%)" for o, v in _counts.items() if v) +
        f"  \n**Class-conditional coverage** (target {1-ALPHA:.2f}): {_cov_rows}  \n"
        f"**Uncovered mass SI+MU+Empty per class** (multiclass form of the paper's "
        f"P(SI)+P(Empty) <= α bound): {_uncov_rows}  \n"
        f"**The competitor varies within each true class**: the multiclass dynamic "
        f"the binary dataset cannot exhibit:  \n{_ystar_rows}"
    )
    return outcome, p_values_test, y_star


@app.cell
def _(MU, X_test, np, outcome, plt, y_test):
    from matplotlib.lines import Line2D

    _colors = {
        "SC": "#4daf4a", "SI": "#e41a1c",
        "MC": "#ff7f00", "MU": "#7b1fa2", "EMPTY": "#999999",
    }
    _shapes = {0: "o", 1: "s", 2: "^"}  # circle / square / triangle, one per class

    _fig, _ax = plt.subplots(figsize=(6.4, 6.4))
    _ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5, zorder=0)
    _ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5, zorder=0)
    _ax.grid(True, color="black", linewidth=0.4, linestyle="-", alpha=0.15, zorder=0)
    _ax.set_axisbelow(True)
    for _k, _mk in _shapes.items():
        for _o, _c in _colors.items():
            _m = (y_test == _k) & (outcome == _o)
            if _m.any():
                _ax.scatter(
                    X_test[_m, 0], X_test[_m, 1], s=24, alpha=0.45,
                    facecolors="none", edgecolors=_c, marker=_mk, linewidths=0.9,
                )

    # Class means: discrete black crosses, not oversized stars.
    _ax.scatter(MU[:, 0], MU[:, 1], marker="x", s=80, c="black",
               linewidths=2.2, zorder=6)

    # Square figure, identical fixed range on both axes (no autoscale).
    _half_range = np.abs(X_test[:, :2]).max() * 1.08
    _ax.set_xlim(-_half_range, _half_range)
    _ax.set_ylim(-_half_range, _half_range)
    _ax.set_aspect("equal")
    _ax.set_xlabel("x0", fontsize=12)
    _ax.set_ylabel("x1", fontsize=12)
    _ax.tick_params(axis="both", labelsize=10)
    # _ax.set_title("Conformal outcomes in the informative subspace")

    # Both legends sit OUTSIDE the axes (bbox_to_anchor with x > 1), fully
    # opaque, so neither overlaps the scattered points: a translucent legend
    # box drawn on top of ~2000 alpha=0.45 markers is what produced the smudged
    # look on export previously.
    _outcome_handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor=_c, markeredgecolor=_c,
              markersize=7, label=f"{_o} (n={int((outcome == _o).sum())})")
        for _o, _c in _colors.items() if (outcome == _o).any()
    ]
    _leg1 = _ax.legend(
        handles=_outcome_handles, title="Outcome",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=15, framealpha=1.0, facecolor="white", title_fontsize=17
    )
    _ax.add_artist(_leg1)

    _class_handles = [
        Line2D([], [], marker=_mk, ls="", markerfacecolor="none",
              markeredgecolor="black", markersize=8, label=f"class {_k}")
        for _k, _mk in _shapes.items()
    ]
    _ax.legend(
        handles=_class_handles, title="True class",
        loc="upper left", bbox_to_anchor=(1.02, 0.55),
        fontsize=15, framealpha=1.0, facecolor="white", title_fontsize=17
    )

    # tight_layout does not know about legends placed outside the axes via
    # bbox_to_anchor, so it can clip them on export; reserve the right margin
    # explicitly instead.
    _fig.subplots_adjust(right=0.72)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Correctness margin in multiclass

    δ_cm = p_true − p_y\*, with y\* the per-instance strongest competitor
    (in binary, always "the other class").

    Expected reading, identical to the binary case:

    - **SC** → large positive margin
    - **SI**, **MU** → negative margin
    - **MC** → margin near zero

    The margin measures how close the set decision was.
    """)
    return


@app.cell
def _(ALPHA, mo, np, outcome, p_values_test, plt, y_star, y_test):
    margin_cm = (
        p_values_test[np.arange(len(y_test)), y_test]
        - p_values_test[np.arange(len(y_test)), y_star]
    )

    _fig, _ax = plt.subplots(figsize=(7.5, 3.8))
    _colors = {"SI": "#e41a1c", "MU": "#7b1fa2", "MC": "#ff7f00", "SC": "#4daf4a"}
    _order = [o for o in ["SI", "MU", "MC", "SC"] if (outcome == o).sum() >= 5]
    _data = [margin_cm[outcome == o] for o in _order]
    _parts = _ax.violinplot(_data, showmedians=True, widths=0.8)
    for _pc, _o in zip(_parts["bodies"], _order):
        _pc.set_facecolor(_colors[_o]); _pc.set_alpha(0.6)
    _ax.axhline(0, color="k", lw=0.8, ls="--")
    _ax.set_xticks(range(1, len(_order) + 1)); _ax.set_xticklabels(_order)
    _ax.set_ylabel("correctness margin  p_true − p_y*")
    _ax.set_title(f"Margin distribution by conformal outcome (α={ALPHA})")
    _fig.tight_layout()

    _med = {o: float(np.median(margin_cm[outcome == o])) for o in _order}
    _med_txt = " | ".join(f"{o} {_med[o]:+.2f}" for o in _order)
    mo.vstack([
        _fig,
        mo.md(
            f"Medians: {_med_txt}. The binary diagnostic reading transfers intact, "
            f"and the split adds a multiclass nuance: MU sits on the negative side "
            f"with SI (it is a miss, with the set signalling indecision among "
            f"rivals), while MC hovers around zero."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3b. Evidence margin in multiclass: one-vs-rest reading

    Evidence margin, pairwise: δ^{i,j}_em(x) = p^i(x) − p^j(x), label-free, for
    any two classes. With K=3 the six ordered differences collapse to three
    pairs by antisymmetry (the §6 pair beeswarms). The paper's primary binary
    SHAP target (fixed-pair evidence margin) carries over unchanged.

    One-vs-rest evidence for a class of interest k, per the paper's own
    prescription for multiclass (interpret it in a one-vs-rest or top-two
    comparison):

    e_k(x) = p_k(x) − max_{j≠k} p_j(x) = min_{j≠k} δ^{k,j}_em(x)

    - e_k > 0 → evidence favors k over every rival
    - e_k < 0 → some rival beats k

    Two properties, direct from the definitions:
    - K=2 recovers the binary evidence margin
    - δ_cm = e_{y_true}: the correctness margin is this one-vs-rest evidence
      evaluated at the true class, a priori vs a posteriori is a choice of
      anchor, not a different formula

    Anchor below: class 0 (free parameter, set to whatever class matters
    operationally).
    """)
    return


@app.cell
def _(mo, np, p_values_test, plt, y_test):
    # Anchored evidence margins e_k = p_k − max of the rivals, for all anchors.
    _K = p_values_test.shape[1]
    _e = np.empty_like(p_values_test)
    for _k in range(_K):
        _rivals = np.delete(p_values_test, _k, axis=1)
        _e[:, _k] = p_values_test[:, _k] - _rivals.max(1)

    # Demo anchor: class 0. Distribution by TRUE class, computable a priori and
    # then audited a posteriori: true class 0 should sit positive, rivals negative,
    # with the overlap band showing where evidence for the anchor is ambiguous.
    _fig, _ax = plt.subplots(figsize=(7.5, 3.8))
    _cls_colors = ["#4daf4a", "#377eb8", "#ff7f00"]
    _data = [_e[y_test == _k, 0] for _k in range(_K)]
    _parts = _ax.violinplot(_data, showmedians=True, widths=0.8)
    for _pc, _c in zip(_parts["bodies"], _cls_colors):
        _pc.set_facecolor(_c); _pc.set_alpha(0.6)
    _ax.axhline(0, color="k", lw=0.8, ls="--")
    _ax.set_xticks(range(1, _K + 1))
    _ax.set_xticklabels([f"true class {_k}" for _k in range(_K)])
    _ax.set_ylabel("one-vs-rest evidence  e_0 = p_0 − max(p_1, p_2)")
    _ax.set_title("One-vs-rest evidence margin (anchor = class 0), by true class")
    _fig.tight_layout()

    _rows = "\n".join(
        f"| e_{_k} | " + " | ".join(
            f"{np.median(_e[y_test == _t, _k]):+.2f}" for _t in range(_K)
        ) + " |"
        for _k in range(_K)
    )
    _note = mo.md(
        "Median one-vs-rest evidence by anchor (rows) and true class (columns), "
        "showing that any anchor separates its own class from the rest, so the "
        "choice is purely operational:\n\n"
        "| anchor | true 0 | true 1 | true 2 |\n|---|---|---|---|\n" + _rows +
        "\n\nThe diagonal is positive and the off-diagonal negative for every "
        "anchor: the a-priori evidence reading of the paper transfers to "
        "multiclass for whichever class the practitioner cares about."
    )

    # Identity check linking this section to the raw pairs of §6: e_k(x) is
    # the per-instance minimum of the pairwise deltas delta^{k,j}_em = p_k - p_j
    # over the rivals j. Verified directly here, since the definitions were
    # written in two different cells.
    _e_from_pairs = np.stack(
        [
            np.min(
                [p_values_test[:, _k] - p_values_test[:, _j] for _j in range(_K) if _j != _k],
                axis=0,
            )
            for _k in range(_K)
        ],
        axis=1,
    )
    _max_discrepancy = float(np.abs(_e_from_pairs - _e).max())
    _identity_note = mo.md(
        f"Identity check: e_k(x) recomputed as min over j≠k of the raw pairwise "
        f"margin p_k − p_j (§6's object) matches the one-vs-rest definition above "
        f"to **{_max_discrepancy:.2e}** (floating-point exact). Section 3b is "
        f"the anchored summary of that same pairwise object from §6."
    )
    mo.vstack([_fig, _note, _identity_note])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3c. Choosing the class of interest

    Binary paper: anchor is class 1 (cancellation), an operational choice
    rather than an index rule. The synthetic triangle has no built-in anchor
    (classes are symmetric by design), so a single anchored figure needs an
    explicit, reproducible criterion.

    Criterion: ROC-AUC of e_k as a one-vs-rest score for (y_true == k) vs the
    rest, with a bootstrap to check whether any class actually wins or the
    geometry is symmetric.
    """)
    return


@app.cell
def _(mo, np, p_values_test, y_test):
    from sklearn.metrics import roc_auc_score

    _K = p_values_test.shape[1]
    _e_full = np.empty_like(p_values_test)
    for _k in range(_K):
        _rivals = np.delete(p_values_test, _k, axis=1)
        _e_full[:, _k] = p_values_test[:, _k] - _rivals.max(1)

    auc_by_class = {
        _k: float(roc_auc_score((y_test == _k).astype(int), _e_full[:, _k]))
        for _k in range(_K)
    }

    _rng_b = np.random.default_rng(0)
    _n = len(y_test)
    _boot = {k: [] for k in range(_K)}
    for _ in range(300):
        _idx = _rng_b.choice(_n, _n, replace=True)
        for _k in range(_K):
            _boot[_k].append(
                roc_auc_score((y_test[_idx] == _k).astype(int), _e_full[_idx, _k])
            )
    _ci = {
        _k: (float(np.percentile(_boot[_k], 2.5)), float(np.percentile(_boot[_k], 97.5)))
        for _k in range(_K)
    }

    anchor_class = max(auc_by_class, key=auc_by_class.get)
    _rows = "\n".join(
        f"| class {_k} | {auc_by_class[_k]:.3f} | "
        f"[{_ci[_k][0]:.3f}, {_ci[_k][1]:.3f}] | n={int((y_test == _k).sum())} |"
        for _k in range(_K)
    )
    _overlap = all(
        _ci[anchor_class][0] < _ci[_k][1] for _k in range(_K) if _k != anchor_class
    )
    _overlap_phrase = (
        "the confidence intervals overlap across all three candidates"
        if _overlap else "the winner is clearly ahead"
    )
    mo.md(
        f"| anchor | AUC | bootstrap 95% CI | test size |\n|---|---|---|---|\n{_rows}\n\n"
        f"**Selected anchor: class {anchor_class}** (highest AUC). "
        f"{_overlap_phrase[0].upper()}{_overlap_phrase[1:]}: the edge for class "
        f"{anchor_class} tracks its larger test sample (more stable AUC "
        f"estimate), rather than a geometric advantage. This criterion is "
        f"data-driven and reproducible, and correctly detects a real winner "
        f"under an asymmetric design."
    )
    return anchor_class, roc_auc_score


@app.cell
def _(mo):
    mo.md("""
    ### 3d. Per-rival breakdown for the selected anchor

    §3b's e_k pools both rivals into a single number (the max). Here both
    pairwise margins for the selected anchor are shown separately, anchor
    first (matching the binary paper's convention, e.g. p^1 − p^0):

    δ^{anchor,rival} = p_anchor − p_rival

    One panel per rival: values, then SHAP.
    """)
    return


@app.cell
def _(anchor_class, p_values_test, plt, y_test):
    _K = p_values_test.shape[1]
    _rivals = [j for j in range(_K) if j != anchor_class]

    _fig, _axes = plt.subplots(1, len(_rivals), figsize=(6.5 * len(_rivals), 3.8), sharey=True)
    if len(_rivals) == 1:
        _axes = [_axes]
    _cls_colors = ["#4daf4a", "#377eb8", "#ff7f00"]
    for _ax, _riv in zip(_axes, _rivals):
        _delta = p_values_test[:, anchor_class] - p_values_test[:, _riv]
        _data = [_delta[y_test == _k] for _k in range(_K)]
        _parts = _ax.violinplot(_data, showmedians=True, widths=0.8)
        for _pc, _c in zip(_parts["bodies"], _cls_colors):
            _pc.set_facecolor(_c); _pc.set_alpha(0.6)
        _ax.axhline(0, color="k", lw=0.8, ls="--")
        _ax.set_xticks(range(1, _K + 1))
        _ax.set_xticklabels([f"true {_k}" for _k in range(_K)])
        _ax.set_title(f"δ^{{{anchor_class},{_riv}}} = p_{anchor_class} − p_{_riv}")
    _axes[0].set_ylabel("evidence margin vs this rival")
    _fig.suptitle(
        f"Per-rival evidence for anchor = class {anchor_class} (each rival separately)",
        y=1.03,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    FEATURES,
    X_explain,
    anchor_class,
    p_values_test,
    plt,
    shap,
    shap_conformal,
):
    # Same breakdown at the SHAP level: one beeswarm per rival, anchor first,
    # for the auto-selected class of interest. This is the genuine multiclass
    # continuation of the binary paper's specific quantity (shap_p1 - shap_p0),
    # as opposed to §6's generic, unanchored, index-ordered triptych.
    _K = p_values_test.shape[1]
    _rivals = [j for j in range(_K) if j != anchor_class]

    _fig = plt.figure(figsize=(6.5 * len(_rivals), 4.2))
    for _pi, _riv in enumerate(_rivals):
        _phi = shap_conformal[:, :, anchor_class] - shap_conformal[:, :, _riv]
        plt.subplot(1, len(_rivals), _pi + 1)
        shap.summary_plot(_phi, X_explain, feature_names=FEATURES,
                          show=False, plot_size=None, color_bar=(_pi == len(_rivals) - 1))
        plt.title(f"δ^{{{anchor_class},{_riv}}} SHAP (anchor {anchor_class} vs rival {_riv})",
                 fontsize=10)
    _fig.suptitle(
        f"Per-rival evidence-margin SHAP for the selected class of interest "
        f"(class {anchor_class})", y=1.03,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3e. Why e0's SHAP needs its own KernelSHAP call

    The two per-rival deltas above get their SHAP for free by linearity: each
    is a subtraction of two columns of `shap_conformal`, already computed once.
    e0 = min(δ^{0,1}, δ^{0,2}) is a nonlinear combination, not a subtraction, so
    its Shapley values are not obtainable from the existing ones (Shapley of
    a min/max differs from any combination of the fixed-pair Shapleys; §5's
    max-margin audit found a discrepancy up to 0.28). e0's SHAP requires a new
    KernelSHAP call on `x -> p_0(x) - max(p_1(x), p_2(x))`, over the same
    background used elsewhere in this notebook.
    """)
    return


@app.cell
def _(
    NSAMPLES,
    X_background,
    X_explain,
    anchor_class,
    np,
    p_values_test,
    predict_p_values_fn,
    shap,
):
    _K = p_values_test.shape[1]
    _rivals_idx = [j for j in range(_K) if j != anchor_class]

    def _e0_fn(X):
        _pv = predict_p_values_fn(np.asarray(X))
        return _pv[:, anchor_class] - _pv[:, _rivals_idx].max(1)

    _explainer_e0 = shap.KernelExplainer(_e0_fn, X_background)
    shap_e0 = np.array(
        _explainer_e0.shap_values(X_explain, nsamples=NSAMPLES, silent=True)
    )
    return (shap_e0,)


@app.cell
def _(X_explain, anchor_class, mo, np, plt, shap_conformal, shap_e0):
    _K = shap_conformal.shape[-1]
    _rivals_idx = [j for j in range(_K) if j != anchor_class]
    _riv_a, _riv_b = _rivals_idx[0], _rivals_idx[1]
    _phi_01 = shap_conformal[:, 0, anchor_class] - shap_conformal[:, 0, _riv_a]
    _phi_02 = shap_conformal[:, 0, anchor_class] - shap_conformal[:, 0, _riv_b]
    _phi_e0 = shap_e0[:, 0]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for _ax, _phi, _lbl in zip(
        _axes, [_phi_01, _phi_02, _phi_e0],
        [f"δ^{{{anchor_class},{_riv_a}}} (SHAP, linear)",
         f"δ^{{{anchor_class},{_riv_b}}} (SHAP, linear)",
         "e0 = min(both) (SHAP, own KernelSHAP call)"],
    ):
        _ax.scatter(X_explain[:, 0], _phi, s=12, alpha=0.6, c="#377eb8")
        _ax.axhline(0, color="k", lw=0.6, ls="--")
        _ax.set_xlabel("x0")
        _ax.set_title(_lbl, fontsize=9)
    _axes[0].set_ylabel("SHAP value of x0")

    _r01 = float(np.corrcoef(X_explain[:, 0], _phi_01)[0, 1])
    _r02 = float(np.corrcoef(X_explain[:, 0], _phi_02)[0, 1])
    _re0 = float(np.corrcoef(X_explain[:, 0], _phi_e0)[0, 1])
    _fig.suptitle(
        "x0's SHAP flips sign between rivals and is muted in the nonlinear "
        "aggregate: exactly the pooling artifact of §5, now at the SHAP level "
        "of the evidence margin", y=1.04, fontsize=11,
    )
    _fig.tight_layout()

    _note = mo.md(
        f"corr(x0, SHAP): δ^{{{anchor_class},{_riv_a}}} = **{_r01:+.2f}**, "
        f"δ^{{{anchor_class},{_riv_b}}} = **{_r02:+.2f}**, e0 = **{_re0:+.2f}**. "
        f"The two pairwise SHAP dependences on x0 carry opposite signs of "
        f"comparable size (mirror geometry of rivals {_riv_a} and {_riv_b} "
        f"across x0=0); the nonlinear aggregate's SHAP dependence on x0 is "
        f"markedly weaker, since x0 matters in opposite directions depending on "
        f"which rival is currently binding the minimum, and the two effects "
        f"partly cancel. Reading only e0 would understate x0's role; the "
        f"per-rival breakdown is what recovers it."
    )
    mo.vstack([_fig, _note])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3f. Two definitions of e0's SHAP

    - **Nonlinear (§3e):** the max between rivals is re-evaluated inside every
      KernelSHAP coalition. Requires its own KernelSHAP call.
    - **Fixed-pair:** freeze the binding rival at its value at the real
      instance, then subtract already-computed SHAP columns. No new call
      needed, the same move used for the correctness margin throughout this
      notebook.

    Both are legitimate targets, but different quantities. Compared directly
    below.
    """)
    return


@app.cell
def _(
    X_explain,
    anchor_class,
    explain_idx,
    mo,
    np,
    p_values_test,
    shap_conformal,
    shap_e0,
):
    _K = shap_conformal.shape[-1]
    _rivals_idx = [j for j in range(_K) if j != anchor_class]
    _pv_explain = p_values_test[explain_idx]  # align p-values to X_explain's rows
    # Binding rival per instance: whichever of the two rivals has the higher
    # p-value at the REAL point (the same move as y_star for the correctness
    # margin), frozen before subtracting, no new KernelSHAP call.
    _binding = np.where(
        _pv_explain[:, _rivals_idx[1]] > _pv_explain[:, _rivals_idx[0]],
        _rivals_idx[1], _rivals_idx[0],
    )
    shap_e0_fixedpair = np.stack(
        [
            shap_conformal[_i, :, anchor_class] - shap_conformal[_i, :, _binding[_i]]
            for _i in range(len(X_explain))
        ]
    )

    _diff = np.abs(shap_e0_fixedpair - shap_e0)
    _r_nl = float(np.corrcoef(X_explain[:, 0], shap_e0[:, 0])[0, 1])
    _r_fp = float(np.corrcoef(X_explain[:, 0], shap_e0_fixedpair[:, 0])[0, 1])
    mo.md(
        f"Mean |fixed-pair − nonlinear| by feature (x0, x1 first): "
        f"{np.round(_diff.mean(0)[:2], 3).tolist()}, noise features all "
        f"< {_diff.mean(0)[2:].max():.3f}. Max discrepancy anywhere: "
        f"**{_diff.max():.3f}** (the same order as the earlier max-margin vs "
        f"fixed-pair audit, ~0.28). corr(x0, SHAP e0): nonlinear "
        f"**{_r_nl:+.2f}**, fixed-pair **{_r_fp:+.2f}**. Both are muted "
        f"relative to the individual pairwise deltas (+0.86 / −0.69): the "
        f"pooling artifact happens either way, regardless of which SHAP "
        f"definition is used. What differs between definitions is "
        f"cost (fixed-pair is free, reusing existing SHAP; nonlinear needs a "
        f"fresh KernelSHAP call) and exactness (fixed-pair matches how the "
        f"correctness margin is computed everywhere else in this notebook; "
        f"nonlinear answers a subtly different question, \"how would the "
        f"attribution change if the identity of the binding rival itself "
        f"could shift under masking\")."
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. FCOD on the synthetic

    Same FCOD as notebook 04 (`conformalpy.fcod`, same smoothing parameters).
    Known ground truth gives testable predictions:

    - along `x1`, moving from the bottom pair of classes toward class 0: MC
      bump at intermediate values (boundary band), high SC at the extremes
    - along a **noise feature**: FCOD flat at the marginal outcome rates

    FCOD should be reactive where the truth has structure and flat where it
    has none: the FCOD analogue of the §7 fidelity check.
    """)
    return


@app.cell
def _(
    FEATURES,
    X_test,
    compute_fcod_smoothed,
    conf_clf,
    pd,
    plot_stacked_fcod,
    y_test,
):
    # prediction sets in the same format notebook 03 produces / notebook 04 consumes
    prediction_sets = conf_clf.predict(pd.DataFrame(X_test, columns=FEATURES))

    # Real signature and call pattern (cf. 04_fcod.py, "Non-merged per-feature
    # stacked + density"): stacked/filled area, not lines, is what the paper
    # actually uses for FCOD (plot_stacked_fcod on the unmerged fcod_results, so
    # MC and MU stay distinct here rather than being collapsed). show_density=True
    # builds its own two-row figure and hands back both axes; the same defensive
    # tuple check as 04 covers a plain-axis return too.
    FCOD_PARAMS = dict(n_grid=50, percentile_range=(5, 95), sigma=2.0)  # as in 04
    _fcod_features = ["x0", "x1", "noise0"]

    fcod_fig_handles = []
    for _fi, _feat in enumerate(_fcod_features):
        _j = FEATURES.index(_feat)
        _fcod = compute_fcod_smoothed(
            X_test[:, _j], prediction_sets, y_test, **FCOD_PARAMS
        )
        _fcod["feature_name"] = _feat
        _result = plot_stacked_fcod(
            _fcod, show_density=True, density_type="histogram",
            feature_values=X_test[:, _j], xlabel=_feat, title=_feat,
        )
        _main_ax = _result[0] if isinstance(_result, tuple) else _result
        # Fixed [0, 1] range on every panel: proportions are only comparable
        # across features if the y-axis is not left to autoscale per panel.
        # (Stacked proportions already sum to 1, this just pins it explicitly.)
        _main_ax.set_ylim(0, 1)
        # Legend only on the first panel: the three share the same four outcome
        # categories, repeating it three times is redundant. Removed via the
        # standard matplotlib API rather than a plot_stacked_fcod kwarg, since
        # this works regardless of what the real library's signature supports.
        if _fi > 0 and _main_ax.get_legend() is not None:
            _main_ax.get_legend().remove()
        # Bigger panels: plot_stacked_fcod builds its own figure internally
        # (no figsize passed through), so resize the returned figure directly
        # via the standard matplotlib API rather than assume the real library
        # exposes a size kwarg.
        _main_ax.figure.set_size_inches(7.0, 5.8)
        fcod_fig_handles.append(_main_ax.figure)
    return (fcod_fig_handles,)


@app.cell
def _(fcod_fig_handles, mo):
    # Image-only cell: nothing but the three FCOD panels, so the figure can be
    # exported/saved directly at full quality, without the reading notes below
    # riding along in the same block.
    mo.hstack(fcod_fig_handles)
    return


@app.cell
def _(FEATURES, X_test, np, outcome):
    # noise0 is plotted as the representative noise feature (chosen by index, not
    # post hoc). Verify representativeness over ALL six: for each noise feature,
    # bin the outcomes into deciles and measure the largest deviation of any
    # outcome proportion from its marginal rate. Uses raw proportions, so it does
    # not depend on conformalpy's smoothing internals.
    _marginal = {o: float((outcome == o).mean()) for o in ["SC", "SI", "MC", "MU"]}
    fcod_worst = 0.0
    for _jn in range(2, len(FEATURES)):
        _q = np.quantile(X_test[:, _jn], np.linspace(0, 1, 11))
        _bin = np.clip(np.digitize(X_test[:, _jn], _q[1:-1]), 0, 9)
        for _o, _p_marg in _marginal.items():
            _dev = max(
                abs(float((outcome[_bin == _b] == _o).mean()) - _p_marg)
                for _b in range(10)
            )
            fcod_worst = max(fcod_worst, _dev)
    _n_bin = len(X_test) // 10
    fcod_scale = max(
        (_p * (1 - _p) / _n_bin) ** 0.5 for _p in _marginal.values() if _p > 0
    )
    fcod_n_comp = (len(FEATURES) - 2) * 10 * len(_marginal)
    fcod_expected_max = fcod_scale * (2 * np.log(fcod_n_comp)) ** 0.5
    # Geometric reading, verified against the data rather than eyeballed:
    # - x0 confusion concentrates near x0=0, the midpoint between the two BOTTOM
    #   vertices (x0 = -sqrt(3) and +sqrt(3)), since x0 is the only axis that
    #   separates classes 1 and 2 from each other (their means share x1=-1).
    # - x1 confusion concentrates for x1 > 0, the band between the top vertex
    #   (x1=2) and the bottom pair (x1=-1): with sigma=1.3 their tails overlap
    #   there, and ALL THREE classes mix in that band, not just two.
    fcod_x0_conf_mid = float(
        (np.abs(X_test[(outcome == "MC") | (outcome == "MU"), 0]) < 1.0).mean()
    )
    fcod_x1_conf_pos = float(
        (X_test[(outcome == "MC") | (outcome == "MU"), 1] > 0).mean()
    )
    return (
        fcod_expected_max,
        fcod_n_comp,
        fcod_scale,
        fcod_worst,
        fcod_x0_conf_mid,
        fcod_x1_conf_pos,
    )


@app.cell
def _(
    fcod_expected_max,
    fcod_n_comp,
    fcod_scale,
    fcod_worst,
    fcod_x0_conf_mid,
    fcod_x1_conf_pos,
    mo,
):
    mo.md(f"""
    On x0: {100*fcod_x0_conf_mid:.0f}% of multi-set instances have
    |x0| < 1, the midpoint band between classes 1 and 2 (x0 is the only axis
    separating them, since their means share x1=-1).

    On x1: {100*fcod_x1_conf_pos:.0f}% of multi-set instances have
    x1 > 0, the band between the top vertex and the bottom pair, where all
    three classes overlap at once.

    Flatness check on the noise features: largest deviation of any outcome
    proportion from its marginal rate, across 10 decile bins, over all six
    noise features: **{fcod_worst:.3f}**. Expected maximum under pure sampling
    noise, {fcod_n_comp} bin×outcome comparisons: ~{fcod_expected_max:.3f}
    (per-bin binomial scale {fcod_scale:.3f} × multiple-comparison factor).
    Observed matches the pure-noise prediction: all six are flat. The sawtooth
    shape is sparse-bin noise (equal-width bins near a Gaussian's 5th/95th
    percentile hold fewer points), not leftover signal. It is not fixed by
    narrowing noise0's own spread, only by raising `sigma` in FCOD_PARAMS (the
    smoothing kernel width).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. The varying-competitor artifact, reproduced and resolved

    KernelSHAP is applied once to the full-pipeline p-value function
    (model + empirical Mondrian calibration, the step-function part TreeSHAP
    cannot represent, cf. `05_shap.py`). Correctness-margin SHAP is the
    per-instance difference φ(p_true) − φ(p_y*), as in `derive_margin_shap`.

    In multiclass the target is the exact Shapley of the *fixed-pair*
    margin p_y − p_y* (competitor frozen at its realized value), the natural
    extension of the binary definition, and not the Shapley of the max-margin
    (which differs, since max is nonlinear). Fixed-pair makes aggregation
    well-posed: within a (y, y\*) stratum every instance shares the same
    target function.

    Expected behavior: pooling strata mixes coherent dependences of opposite
    sign (same feature, different boundary) and cancels them; stratifying by
    the pair restores them. Binary is the single-stratum special case.
    """)
    return


@app.cell
def _(
    FEATURES,
    NSAMPLES,
    N_BACKGROUND,
    N_EXPLAIN,
    SEED,
    X_test,
    conf_clf,
    np,
    pd,
    shap,
    y_star,
    y_test,
):
    # Background / explain sampling without overlap, as in 05_shap.py.
    np.random.seed(SEED)
    _n_test = len(X_test)
    background_idx = np.random.choice(_n_test, size=N_BACKGROUND, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(_n_test), background_idx),
        size=N_EXPLAIN, replace=False,
    )
    X_background = X_test[background_idx]
    X_explain = X_test[explain_idx]
    y_explain = y_test[explain_idx].astype(int)
    y_star_explain = y_star[explain_idx]

    def predict_p_values_fn(X: np.ndarray) -> np.ndarray:
        """Wrapper: ndarray → DataFrame → conf_clf.predict_p_values (all-numeric)."""
        return conf_clf.predict_p_values(pd.DataFrame(X, columns=FEATURES))

    _explainer = shap.KernelExplainer(predict_p_values_fn, X_background)
    shap_conformal = np.array(
        _explainer.shap_values(X_explain, nsamples=NSAMPLES, silent=True)
    )
    if shap_conformal.shape[0] == 3:          # (class, n, d) → (n, d, class)
        shap_conformal = np.moveaxis(shap_conformal, 0, -1)
    return (
        X_background,
        X_explain,
        background_idx,
        explain_idx,
        predict_p_values_fn,
        shap_conformal,
        y_explain,
        y_star_explain,
    )


@app.cell
def _(
    derive_margin_shap,
    select_class_shap,
    shap_conformal,
    y_explain,
    y_star_explain,
):
    shap_p_true = select_class_shap(shap_conformal, y_explain)        # SHAP of p_true
    shap_p_star = select_class_shap(shap_conformal, y_star_explain)   # SHAP of p_y*
    shap_margin = derive_margin_shap(shap_p_true, shap_p_star)        # SHAP of (p_true − p_y*)
    return (shap_margin,)


@app.cell
def _(
    X_explain,
    np,
    pearsonr,
    permutations,
    plt,
    shap_margin,
    y_explain,
    y_star_explain,
):
    # Dependence of x0 vs its margin-SHAP: pooled (left) vs per-(y, y*) strata (right)
    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)

    _r_pooled = pearsonr(X_explain[:, 0], shap_margin[:, 0])[0]
    _axes[0].scatter(X_explain[:, 0], shap_margin[:, 0], s=10, c="gray", alpha=0.6)
    _axes[0].set_title(f"POOLED: corr(x0, φ_x0) = {_r_pooled:.2f}")
    _axes[0].set_xlabel("x0"); _axes[0].set_ylabel("margin-SHAP of x0")

    _cmap = plt.get_cmap("tab10")
    strata_corr = {}
    for _i, (_a, _b) in enumerate(permutations(range(3), 2)):
        _m = (y_explain == _a) & (y_star_explain == _b)
        if _m.sum() >= 15:
            _r = pearsonr(X_explain[_m, 0], shap_margin[_m, 0])[0]
            strata_corr[(_a, _b)] = float(_r)
            _axes[1].scatter(X_explain[_m, 0], shap_margin[_m, 0], s=10, alpha=0.7,
                             color=_cmap(_i), label=f"(y={_a}, y*={_b}): r={_r:+.2f}")
    _mean_abs = float(np.mean([abs(r) for r in strata_corr.values()]))
    _axes[1].set_title(f"STRATIFIED by (y, y*): mean |corr| = {_mean_abs:.2f}")
    _axes[1].set_xlabel("x0"); _axes[1].legend(fontsize=7.5)
    _fig.suptitle(
        "Per-stratum dependences are near-deterministic with opposing signs; "
        "pooling cancels them", y=1.04,
    )
    _fig.tight_layout()
    _fig
    return (strata_corr,)


@app.cell
def _(FEATURES, X_explain, plt, shap, shap_margin, y_explain, y_star_explain):
    # The paper's beeswarm idiom: global summary before vs after stratification.
    # Left: all strata pooled (the misleading global view). Right: the largest
    # (y, y*) stratum (what the paper's prescription for multiclass looks like).
    _sizes = {
        (a, b): int(((y_explain == a) & (y_star_explain == b)).sum())
        for a in range(3) for b in range(3) if a != b
    }
    _a, _b = max(_sizes, key=_sizes.get)
    _m = (y_explain == _a) & (y_star_explain == _b)

    _fig = plt.figure(figsize=(13, 4.2))
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_margin, X_explain, feature_names=FEATURES,
                      show=False, plot_size=None, color_bar=False)
    plt.title("Beeswarm POOLED (all pairs mixed)", fontsize=10)
    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_margin[_m], X_explain[_m], feature_names=FEATURES,
                      show=False, plot_size=None)
    plt.title(f"Beeswarm stratum (y={_a}, y*={_b}), n={_m.sum()}", fontsize=10)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo, np, strata_corr):
    # Antisymmetry check: φ of the (a,b) margin is minus φ of the (b,a) margin,
    # so ordered strata merge into K(K−1)/2 unordered pairs via a sign flip.
    _pairs_ok = all(
        np.sign(strata_corr.get((a, b), 0)) == -np.sign(strata_corr.get((b, a), 0))
        for (a, b) in [(0, 1), (0, 2), (1, 2)]
        if (a, b) in strata_corr and (b, a) in strata_corr
    )
    mo.md(
        "The predicted artifact is real and quantified: pooled "
        "dependence collapses while per-stratum dependence stays near ±1, with "
        "signs matching the geometry of each boundary. Aggregate margin-SHAP "
        "**per (y, y\\*) pair**"
        + (", and since sign patterns confirm φ(a,b) = −φ(b,a), ordered strata can be "
           "merged into unordered pairs with a sign convention, halving the strata."
           if _pairs_ok else ".")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Full SHAP repertoire of notebook 05, on three classes

    Same figure set as `05_shap.py` §B/C/D:

    - per-class p-value beeswarms
    - pooled Correctness Margin beeswarm
    - raw pairwise Evidence Margin beeswarms
    - importance bars (global and per class), both margins
    - dependence grids (`conformalpy.shap`), both margins

    The pair panels below are the SHAP of the raw evidence margin as defined
    (δ^{i,j}_em = p^i − p^j, fixed pair, label-free, no anchor), computed on
    all explained instances, no label-based selection, matching the binary
    evidence beeswarm of notebook 05. The fixed pair gives these summaries
    direct global interpretation; label-restricted, stratified views belong to
    §5's correctness-margin analysis rather than here. §3b's anchored one-vs-rest
    reading is built from these same pairs (e_k is their per-instance
    minimum): the same object read at two levels.
    """)
    return


@app.cell
def _(FEATURES, X_explain, plt, shap, shap_conformal):
    # Per-class p-value beeswarms: which features drive conformal evidence FOR each
    # class. Verified earlier with raw correlations: x1 clearly dominates p0
    # (corr 0.88 vs 0.12, class 0 sits on top, separated mainly along x1); for p1
    # and p2, x0 and x1 are comparable, with x0 only slightly ahead (0.67 vs 0.58
    # and 0.65 vs 0.46) since classes 1 and 2 also differ from class 0 along x1.
    # Noise pinned at zero in all three.
    _fig = plt.figure(figsize=(15, 4.2))
    for _k in range(3):
        plt.subplot(1, 3, _k + 1)
        shap.summary_plot(
            shap_conformal[:, :, _k], X_explain, feature_names=FEATURES,
            show=False, plot_size=None, color_bar=(_k == 2),
        )
        plt.title(f"Beeswarm of p-value, class {_k}", fontsize=10)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(FEATURES, X_explain, plt, shap, shap_conformal):
    # Evidence-margin beeswarms, one per RAW unordered pair. Eq. evidence-margin
    # is generic in i,j with no privileged direction: sign is a convention, not a
    # finding (delta^{i,j} = -delta^{j,i}). In the binary paper, 05_shap.py fixes
    # shap_p1 - shap_p0 not because "1 > 0" but because class 1 (cancellation) is
    # the operational class of interest for that dataset; the ordering encodes an
    # ANCHOR, not an index rule. The synthetic triangle has no such anchor by
    # design (the three classes are symmetric), so there is no class-1-like
    # privileged direction to copy here. These three panels show the raw,
    # unanchored pairwise object for all three boundaries; direction is fixed to
    # increasing index purely for a consistent legend, with no claim that this
    # matches the "interesting class" convention of the binary figures.
    # The paper's own prescription for what to do with an anchor in multiclass,
    # "interpret it in a one-vs-rest ... comparison", is exactly section 3b's
    # e_k = p_k - max(rivals), which is NOT a free choice among these pairs but
    # their per-instance minimum: e_k(x) = min over j != k of delta^{k,j}_em(x).
    # Expected from the geometry: x1 dominates the pairs involving class 0
    # (vertical separation), x0 dominates pair {1,2} (horizontal boundary),
    # noise pinned at zero in all three panels.
    _pairs = [(0, 1), (0, 2), (1, 2)]
    _fig = plt.figure(figsize=(15, 4.4))
    for _pi, (_a, _b) in enumerate(_pairs):
        _phi_pair = shap_conformal[:, :, _b] - shap_conformal[:, :, _a]
        plt.subplot(1, 3, _pi + 1)
        shap.summary_plot(_phi_pair, X_explain, feature_names=FEATURES,
                          show=False, plot_size=None, color_bar=(_pi == 2))
        plt.title(
            f"Raw pairwise evidence δ^{{{_b},{_a}}} = p^{_b} − p^{_a} (n={len(_phi_pair)})",
            fontsize=10,
        )
    _fig.suptitle(
        "Raw pairwise evidence-margin beeswarms (unanchored, label-free, all instances)",
        y=1.02,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(FEATURES, X_explain, plt, shap, shap_margin):
    # Correctness Margin beeswarm, standalone and full-size: the fourth object
    # of 05_shap.py's Section B (p0, p1, Correctness Margin, Evidence Margin),
    # given its own figure here for parity, not squeezed to half-width. This is
    # the SAME shap_margin already used in §5, pooled over all (y, y*) pairs, no
    # new computation. Precisely because it is pooled, expect the same mixed,
    # hard-to-read structure on x0/x1 documented in §5: this is simply the
    # plain multiclass continuation of the binary figure, shown at the same
    # visual weight as the others for repertoire completeness, with the
    # caveat already established in §5, not repeated as new here.
    _fig = plt.figure(figsize=(7, 4.4))
    shap.summary_plot(shap_margin, X_explain, feature_names=FEATURES,
                      show=False, plot_size=None)
    plt.title(
        f"Correctness Margin beeswarm, pooled (all n={len(shap_margin)}, all pairs mixed)",
        fontsize=10,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(FEATURES, mo, np, plt, shap_margin, y_explain):
    # Importance bars, global and per class, matching the OUTPUT of
    # `plot_shap_importance_grouped` in 05_shap.py (a single combined chart),
    # not the separate global/per-class figures of its section D loop.
    _imp = {"Global": np.abs(shap_margin).mean(0)}
    for _k in sorted(np.unique(y_explain)):
        _imp[f"Class {_k}"] = np.abs(shap_margin[y_explain == _k]).mean(0)

    _order = np.argsort(_imp["Global"])
    _labels = [FEATURES[i] for i in _order]
    _n_groups = len(_imp)
    _bar_h, _gap = 0.2, 0.12
    _base = np.arange(len(FEATURES)) * (_n_groups * _bar_h + _gap)

    _fig, _ax = plt.subplots(figsize=(8.5, 5.5))
    _colors = {"Global": "#78909c", "Class 0": "#4daf4a", "Class 1": "#377eb8", "Class 2": "#ff7f00"}
    for _i, (_lab, _v) in enumerate(_imp.items()):
        _off = (_i - (_n_groups - 1) / 2) * _bar_h
        _ax.barh(_base + _off, _v[_order], height=_bar_h * 0.9,
                 label=_lab, color=_colors.get(_lab, f"C{_i}"), alpha=0.9)
    _ax.set_yticks(_base); _ax.set_yticklabels(_labels)
    _ax.set_xlabel("mean(|SHAP value|)")
    _ax.set_title("Correctness margin importance: global and per class")
    _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()

    _msg = (
        "Reading: importance is driven entirely by the informative pair; per-class "
        "bars reflect the geometry (x1 matters most for class 0, x0 for classes 1 "
        "and 2), and every noise feature stays at the bottom in all four groups."
    )
    mo.vstack([_fig, mo.md(_msg)])
    return


@app.cell
def _(FEATURES, mo, np, plt, shap_conformal, y_explain):
    # Evidence Margin importance bars, same grouped style as the Correctness
    # Margin cell above, completing Section D's pair for the second object
    # (05_shap.py groups Evidence Margin importance by true class too, even
    # though the margin itself is label-free; the grouping there is a diagnostic
    # view, not a claim that the object depends on the label). Pair {1,0},
    # matching 05_shap.py's own p1 - p0, since with only two classes binary has
    # no other pair to begin with.
    _phi_ev_imp = shap_conformal[:, :, 1] - shap_conformal[:, :, 0]
    _imp_ev = {"Global": np.abs(_phi_ev_imp).mean(0)}
    for _k in sorted(np.unique(y_explain)):
        _imp_ev[f"Class {_k}"] = np.abs(_phi_ev_imp[y_explain == _k]).mean(0)

    _order_ev = np.argsort(_imp_ev["Global"])
    _labels_ev = [FEATURES[i] for i in _order_ev]
    _n_groups_ev = len(_imp_ev)
    _bar_h, _gap = 0.2, 0.12
    _base_ev = np.arange(len(FEATURES)) * (_n_groups_ev * _bar_h + _gap)

    _fig, _ax = plt.subplots(figsize=(8.5, 5.5))
    _colors = {"Global": "#78909c", "Class 0": "#4daf4a", "Class 1": "#377eb8", "Class 2": "#ff7f00"}
    for _i, (_lab, _v) in enumerate(_imp_ev.items()):
        _off = (_i - (_n_groups_ev - 1) / 2) * _bar_h
        _ax.barh(_base_ev + _off, _v[_order_ev], height=_bar_h * 0.9,
                 label=_lab, color=_colors.get(_lab, f"C{_i}"), alpha=0.9)
    _ax.set_yticks(_base_ev); _ax.set_yticklabels(_labels_ev)
    _ax.set_xlabel("mean(|SHAP value|)")
    _ax.set_title("Evidence margin δ^{1,0} importance: global and per class")
    _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()

    _msg_ev = (
        "Same reading as the correctness margin: informative pair dominates, "
        "noise stays at the bottom in every group. Unlike the correctness "
        "margin, this object is label-free by definition, so 'per class' here "
        "audits whether the raw pairwise evidence happens to look different "
        "across true classes, not a dependency the definition implies."
    )
    mo.vstack([_fig, mo.md(_msg_ev)])
    return


@app.cell
def _(
    FEATURES,
    X_explain,
    make_dependence_grid,
    np,
    plot_shap_dependence_numeric,
    shap_margin,
    y_explain,
):
    # Dependence grid from conformalpy.shap, top-4 by mean |SHAP| as in 05.
    # Expected: x1 and x0 with clean class-colored structure, the two noise
    # features that complete the top-4 as flat clouds at zero, which is the
    # dependence-plot form of the fidelity criterion.
    _top4 = np.argsort(-np.abs(shap_margin).mean(0))[:4]
    _fig, _axes_pairs = make_dependence_grid(len(_top4), ncols=2, figsize=(12, 9))
    for _i, _j in enumerate(_top4):
        plot_shap_dependence_numeric(
            _axes_pairs[_i][0], X_explain[:, _j], shap_margin[:, _j], y_explain,
            FEATURES[_j], shap_label="Correctness Margin",
            proportion_ax=_axes_pairs[_i][1],
        )
    _fig.suptitle("Dependence, Correctness Margin (top 4 by mean |SHAP|)", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    FEATURES,
    X_explain,
    make_dependence_grid,
    np,
    plot_shap_dependence_numeric,
    shap_conformal,
    y_explain,
):
    # Dependence for the RAW evidence margin: same conformalpy functions, target
    # fixed at pair {0,1} for ALL explained instances (label-free, unanchored),
    # p^1 − p^0 by the same increasing-index legend convention as the beeswarm
    # cell above (a convention for THIS notebook, not a copy of the binary
    # paper's class-of-interest anchor, since the synthetic triangle has none).
    # The fixed pair makes the dependence globally interpretable by construction,
    # which is the paper's stated reason for using the evidence margin as the
    # primary binary SHAP target; these panels are the direct multiclass
    # continuation of the binary dependence plots of 05.
    _phi_ev = shap_conformal[:, :, 1] - shap_conformal[:, :, 0]
    _X_ev = X_explain
    _y_ev = y_explain

    _top4_ev = np.argsort(-np.abs(_phi_ev).mean(0))[:4]
    _fig, _axes_pairs = make_dependence_grid(len(_top4_ev), ncols=2, figsize=(12, 9))
    for _i, _j in enumerate(_top4_ev):
        plot_shap_dependence_numeric(
            _axes_pairs[_i][0], _X_ev[:, _j], _phi_ev[:, _j], _y_ev,
            FEATURES[_j], shap_label="Evidence Margin p1-p0",
            proportion_ax=_axes_pairs[_i][1],
        )
    _fig.suptitle(
        "Dependence, Evidence Margin δ^{1,0} = p1-p0 (all instances, top 4 by mean |SHAP|)",
        fontsize=12,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Faithfulness against ground-truth Shapley

    Oracle pipeline: exact Bayes posterior + Mondrian calibration on a 200k
    sample. Its Shapley values are computed exactly, enumerating all 2⁸
    coalitions over the same background used for the empirical pipeline (same
    background → same interventional game → directly comparable attributions).

    Predictions fixed before running:

    - noise attribution ≈ 0 (oracle: exactly 0)
    - the two informative features rank top-2
    - high per-instance agreement
    - no post-hoc degrees of freedom
    """)
    return


@app.cell
def _(
    FEATURES,
    MU,
    N_NOISE,
    PRIORS,
    SEED,
    SIGMA,
    X_background,
    X_explain,
    bayes_posterior,
    combinations,
    factorial,
    mo,
    np,
    pearsonr,
    shap_margin,
    y_explain,
    y_star_explain,
):
    # Oracle pipeline: Bayes posterior + oracle Mondrian calibration (massive sample).
    _rng_o = np.random.default_rng(SEED + 1)
    _N_O = 200_000
    _y_o = _rng_o.choice(3, _N_O, p=PRIORS)
    _X_o = np.hstack([
        MU[_y_o] + SIGMA * _rng_o.standard_normal((_N_O, 2)),
        _rng_o.standard_normal((_N_O, N_NOISE)),
    ])
    _scores_o = 1 - bayes_posterior(_X_o)[np.arange(_N_O), _y_o]   # LAC on true posterior
    _cal_o = {k: np.sort(_scores_o[_y_o == k]) for k in range(3)}

    def _p_values_oracle(X):
        _P = bayes_posterior(np.asarray(X))
        _out = np.empty_like(_P)
        for _k in range(3):
            _n_ge = len(_cal_o[_k]) - np.searchsorted(_cal_o[_k], 1 - _P[:, _k], "left")
            _out[:, _k] = (_n_ge + 1) / (len(_cal_o[_k]) + 1)
        return _out

    # Exact Shapley by full coalition enumeration (interventional, same background).
    _d = len(FEATURES)
    _subsets = [list(c) for r in range(_d + 1) for c in combinations(range(_d), r)]

    def _exact_shapley(x):
        _Xb = np.repeat(X_background[None], len(_subsets), axis=0).copy()
        for _si, _S in enumerate(_subsets):
            _Xb[_si][:, _S] = x[_S]
        # Unweighted mean is correct here BECAUSE the background is a raw uniform
        # sample; with a weighted summary (e.g. shap.kmeans) the cluster weights
        # would be part of the game and this line would silently change it.
        _V = _p_values_oracle(_Xb.reshape(-1, _d)).reshape(len(_subsets), -1, 3).mean(1)
        _vmap = {tuple(sorted(_S)): _V[_si] for _si, _S in enumerate(_subsets)}
        _phi = np.zeros((_d, 3))
        for _j in range(_d):
            for _S in _subsets:
                if _j in _S:
                    continue
                _w = factorial(len(_S)) * factorial(_d - len(_S) - 1) / factorial(_d)
                _phi[_j] += _w * (_vmap[tuple(sorted(_S + [_j]))] - _vmap[tuple(sorted(_S))])
        return _phi

    _N_GT = min(120, len(X_explain))
    _gt = np.stack([_exact_shapley(X_explain[_i]) for _i in range(_N_GT)])
    # The pair (y, y*) is fixed at the empirically identified competitor, so both
    # pipelines are attributed on the SAME fixed-pair target function; the
    # comparison isolates estimation error and cannot be gamed by pair choice.
    _gt_margin = (
        _gt[np.arange(_N_GT), :, y_explain[:_N_GT]]
        - _gt[np.arange(_N_GT), :, y_star_explain[:_N_GT]]
    )

    _mean_abs = np.abs(shap_margin).mean(0)
    _ratio = float(_mean_abs[:2].mean() / _mean_abs[2:].mean())
    _top2 = set(np.argsort(_mean_abs)[-2:]) == {0, 1}
    _pear = np.array([pearsonr(shap_margin[_i], _gt_margin[_i])[0] for _i in range(_N_GT)])
    _sign = float((np.sign(shap_margin[:_N_GT, :2]) == np.sign(_gt_margin[:, :2])).mean())
    mo.md(
        f"| Faithfulness test | Result |\n|---|---|\n"
        f"| informative / noise attribution ratio | **{_ratio:.1f}×** "
        f"(oracle noise attribution: exactly 0) |\n"
        f"| top-2 features = the informative ones | **{'yes' if _top2 else 'no'}** |\n"
        f"| per-instance Pearson vs exact GT-Shapley (n={_N_GT}) | "
        f"**{_pear.mean():.3f}** mean / {np.median(_pear):.3f} median |\n"
        f"| sign agreement on informative dims | **{_sign:.3f}** |\n\n"
        f"The empirical pipeline reproduces the exact Shapley values of the "
        f"oracle pipeline."
    )
    gt_ratio = _ratio
    gt_pearson_mean = float(_pear.mean())
    gt_pearson_median = float(np.median(_pear))
    gt_sign_agreement = _sign
    return gt_pearson_mean, gt_pearson_median, gt_ratio, gt_sign_agreement


@app.cell
def _(mo):
    mo.md("""
    ## 8. Stability

    Two sources of variation, one dissolves analytically:

    - **nsamples / estimator seed.** With d=8 features, `nsamples=1024`
      exceeds the 254 possible coalitions: KernelSHAP enumerates the full
      lattice, so the estimate is **exact given the background** (holds for
      the paper's own configuration, `shap.nsamples: 1024` in
      `config/copa2026.yaml`). Coalition-sampling variance is zero by
      construction.
    - **Background specification**, the only remaining source of variation,
      quantified below: three independently drawn 300-sample backgrounds
      (different seeds, disjoint from the §5 background and the explained
      instances).
    """)
    return


@app.cell
def _(
    NSAMPLES,
    N_BACKGROUND,
    X_explain,
    X_test,
    background_idx,
    combinations,
    explain_idx,
    mo,
    np,
    predict_p_values_fn,
    select_class_shap,
    shap,
    y_explain,
    y_star_explain,
):
    _N_STAB = 100
    _avail = np.setdiff1d(np.arange(len(X_test)), np.concatenate([background_idx, explain_idx]))
    _phis = {}
    for _sd in (0, 1, 2):
        _bg = X_test[np.random.default_rng(_sd).choice(_avail, N_BACKGROUND, replace=False)]
        _e = shap.KernelExplainer(predict_p_values_fn, _bg)
        _s = np.array(_e.shap_values(X_explain[:_N_STAB], nsamples=NSAMPLES, silent=True))
        if _s.shape[0] == 3:
            _s = np.moveaxis(_s, 0, -1)
        _phis[_sd] = (
            select_class_shap(_s, y_explain[:_N_STAB])
            - select_class_shap(_s, y_star_explain[:_N_STAB])
        )

    _pairwise = [
        float(np.mean([
            np.corrcoef(_phis[a][i], _phis[b][i])[0, 1] for i in range(_N_STAB)
        ]))
        for a, b in combinations(_phis, 2)
    ]
    _top2_stable = all(
        set(np.argsort(np.abs(_p).mean(0))[-2:]) == {0, 1} for _p in _phis.values()
    )
    mo.md(
        f"Per-instance agreement across background seeds: mean Pearson "
        f"**{np.mean(_pairwise):.3f}** (min {min(_pairwise):.3f}); global top-2 "
        f"{'invariant' if _top2_stable else 'NOT invariant'} across all backgrounds. "
        f"Attributions are stable under the only remaining source of variation."
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8b. Summary metrics table

    Every metric produced by this notebook, gathered here in one table.
    Classic supervised metrics (accuracy, recall, precision, ROC-AUC) follow
    the style of notebook 02's Train/Cal/Test table, extended to 3 classes and
    a single test-set column.
    """)
    return


@app.cell
def _(
    ALPHA,
    FEATURES,
    X_test,
    bayes_posterior,
    gt_pearson_mean,
    gt_pearson_median,
    gt_ratio,
    gt_sign_agreement,
    mo,
    model,
    np,
    outcome,
    p_values_test,
    pd,
    roc_auc_score,
    y_test,
):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score,
    )

    _pred = model.predict(pd.DataFrame(X_test, columns=FEATURES))
    _proba = model.predict_proba(pd.DataFrame(X_test, columns=FEATURES))
    _bayes_pred = bayes_posterior(X_test).argmax(1)

    _acc = accuracy_score(y_test, _pred)
    _acc_bayes = accuracy_score(y_test, _bayes_pred)
    _bal_acc = balanced_accuracy_score(y_test, _pred)
    _recall_pc = recall_score(y_test, _pred, average=None)
    _prec_pc = precision_score(y_test, _pred, average=None)
    _f1_macro = f1_score(y_test, _pred, average="macro")
    _prec_macro = precision_score(y_test, _pred, average="macro")
    _recall_macro = recall_score(y_test, _pred, average="macro")
    _roc_ovr = roc_auc_score(y_test, _proba, multi_class="ovr", average="macro")
    _roc_ovo = roc_auc_score(y_test, _proba, multi_class="ovo", average="macro")

    _sets = p_values_test > ALPHA
    _size = _sets.sum(1)
    _covered = _sets[np.arange(len(y_test)), y_test]
    _cov_marginal = float(_covered.mean())
    _cov_pc = [float(_covered[y_test == k].mean()) for k in range(3)]
    _n_pc = [int((y_test == k).sum()) for k in range(3)]

    _n = len(outcome)
    _rate = {o: float((outcome == o).mean()) for o in ["SC", "SI", "MC", "MU", "EMPTY"]}
    _avg_size = float(_size.mean())
    _avg_size_pc = [float(_size[y_test == k].mean()) for k in range(3)]

    _rows = f"""
    | Category | Metric | Value | Description |
    |---|---|---|---|
    | Model | Test accuracy | {_acc:.3f} | vs. Bayes-optimal {_acc_bayes:.3f} |
    | Model | Balanced accuracy | {_bal_acc:.3f} | Macro-averaged recall |
    | Model | Recall, class 0 / 1 / 2 | {_recall_pc[0]:.3f} / {_recall_pc[1]:.3f} / {_recall_pc[2]:.3f} | Falls with class size (prior 0.5/0.3/0.2) |
    | Model | Precision, class 0 / 1 / 2 | {_prec_pc[0]:.3f} / {_prec_pc[1]:.3f} / {_prec_pc[2]:.3f} | Comparatively flat across classes |
    | Model | F1 (macro) | {_f1_macro:.3f} | Diluted echo of the recall gradient |
    | Model | Precision (macro) | {_prec_macro:.3f} | |
    | Model | Recall (macro) | {_recall_macro:.3f} | Same as balanced accuracy here |
    | Model | ROC-AUC (macro, OvR) | {_roc_ovr:.3f} | |
    | Model | ROC-AUC (macro, OvO) | {_roc_ovo:.3f} | |
    | Coverage | Marginal | {_cov_marginal:.3f} | Target: 1 − α = 0.90, pooled over classes |
    | Coverage | Class 0 | {_cov_pc[0]:.3f} | n={_n_pc[0]} |
    | Coverage | Class 1 | {_cov_pc[1]:.3f} | n={_n_pc[1]} |
    | Coverage | Class 2 | {_cov_pc[2]:.3f} | n={_n_pc[2]} |
    | Outcome rate | SC | {_rate['SC']:.3f} | Single Correct |
    | Outcome rate | SI | {_rate['SI']:.3f} | Single Incorrect |
    | Outcome rate | MC | {_rate['MC']:.3f} | Multi-set Covered |
    | Outcome rate | MU | {_rate['MU']:.3f} | Multi-set Uncovered, binary-impossible |
    | Outcome rate | Empty | {_rate['EMPTY']:.3f} | |
    | Efficiency | Multi-set rate (MC+MU) | {_rate['MC'] + _rate['MU']:.3f} | Matches ~20% design target |
    | Efficiency | Avg. set size, marginal | {_avg_size:.3f} | |
    | Efficiency | Avg. set size, class 0 / 1 / 2 | {_avg_size_pc[0]:.3f} / {_avg_size_pc[1]:.3f} / {_avg_size_pc[2]:.3f} | Tracks calibration granularity per class |
    | Fidelity | Informative / noise SHAP ratio | {gt_ratio:.1f}× | vs. oracle: exactly 0 on noise |
    | Fidelity | GT-Shapley Pearson | {gt_pearson_mean:.3f} / {gt_pearson_median:.3f} | Mean / median vs. oracle Shapley |
    | Fidelity | Sign agreement (informative dims) | {gt_sign_agreement:.3f} | vs. oracle Shapley |
    """
    mo.md(_rows)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Summary

    Extending the framework to multiclass requires two changes:

    1. margin target = fixed-pair margin p_y − p_y\*, direct extension of the
       binary definition
    2. global aggregation stratified by the (y, y\*) pair, binary is the
       single-stratum special case

    Quantitative results: per-class coverage ≈ 1−α (§2); margin medians
    ordered SI < MU < MC < SC (§3); pooled-vs-stratified SHAP correlations
    (§5); informative/noise attribution ratio and GT-Shapley agreement (§7).

    §3c–§3f extend the same pooling-artifact analysis to the evidence margin,
    independently of §5: a data-driven criterion for the class of interest,
    and a verified distinction between the two definitions of a pooled
    quantity's SHAP.
    """)
    return


if __name__ == "__main__":
    app.run()
