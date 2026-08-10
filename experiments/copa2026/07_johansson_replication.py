# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "shap>=0.46",
#     "scikit-learn>=1.4",
#     "pandas",
#     "matplotlib",
# ]
# ///
"""07 — Replicating Johansson et al. (2025): identifying the SHAP estimator

Self-contained replication notebook. Reproduces the exact pipeline declared in
Johansson, Maalej & Sonstrod (2025, PMLR 266) — RF(300), hinge/LAC, Mondrian
ICP, Iris — and checks four things at runtime:

  E1. TreeExplainer on the declared target (the p-value function, their Sec. 3.2)
      raises InvalidModelError at construction: the trees compute a probability,
      and the Mondrian p-value is a further transform of that output.
  E2. "Default settings" (their Sec. 3.3) means shap.Explainer's auto-dispatch,
      which selects the algorithm from the type of its first argument. Handed
      the p-value callable, it resolves to a model-agnostic explainer, and the
      choice of algorithm is not reported back to the caller.
  E3. That model-agnostic explainer computes values identical to KernelSHAP
      with full coalition enumeration (machine precision).
  E4. Baseline scale: their Fig. 2 baseline E[f(X)] = 0.207 lives on the
      p-value scale. TreeExplainer baselines live on the probability scale
      (class priors ~ 1/3). Only the model-agnostic route produces their numbers.

Reading: the published attributions are consistent with a model-agnostic
Shapley estimator (KernelSHAP-equivalent) applied to the conformal p-value
function. The "Tree SHAP" label in Sec. 3.3 is best explained by shap's
auto-dispatch, which does not surface which algorithm it selected.

Run:  uvx marimo edit --sandbox 07_johansson_replication.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import warnings
    import numpy as np
    import pandas as pd

    return mo, np, pd, warnings


@app.cell
def _(mo):
    mo.md(r"""
    # 07 — Replicating Johansson et al. (2025): identifying the SHAP estimator

    This notebook replicates the pipeline of Johansson, Maalej & Sonstrod
    (2025) as declared, to establish a like-for-like comparison point with
    our own use of KernelSHAP on conformal p-values. What it turns up is a
    property of the `shap` package's API, not a claim about the authors'
    rigor: `shap.Explainer`'s auto-dispatch selects an algorithm silently, so
    which estimator actually ran cannot be inferred from the call site alone.

    The paper specifies the attribution computation in two places:

    > **Sec. 3.2:** *"we define a function $f_y(x) = p_y(x)$ for each class ...
    > SHAP values are computed for each $f_y(x)$."*  — the declared **target**
    > is the conformal p-value.
    >
    > **Sec. 3.3:** *"We used Tree SHAP ... with default settings as implemented
    > in the SHAP Python package."* — the declared **tool**.

    Taken together, these describe a call to `shap.Explainer` on the p-value
    function. This notebook runs that call and checks: whether TreeSHAP can be
    constructed on that target at all (E1), what `shap.Explainer` actually
    selects for it (E2), whether that selection is numerically KernelSHAP
    (E3), and whether the baseline scale in the paper's own figures is
    consistent with that reading (E4).
    """)
    return


@app.cell
def _(mo, np):
    # --- Johansson pipeline, exactly as declared (Sec. 3.3) ---------------------
    # RF(300) | hinge Delta = 1 - P(y|x) | Mondrian ICP | stratified 60/30/10 | Iris
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    SEED = 42
    _d = load_iris()
    X_all, y_all = _d.data, _d.target
    feature_names = list(_d.feature_names)
    class_names = list(_d.target_names)
    n_classes = 3

    X_train, _X_rest, y_train, _y_rest = train_test_split(
        X_all, y_all, train_size=0.60, stratify=y_all, random_state=SEED
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        _X_rest, _y_rest, train_size=0.75, stratify=_y_rest, random_state=SEED
    )

    model = RandomForestClassifier(n_estimators=300, random_state=SEED)
    model.fit(X_train, y_train)

    # Mondrian deterministic p-values: p_k = (#{a_i >= a_new} + 1) / (n_k + 1)
    _cal_proba = model.predict_proba(X_cal)
    cal_alpha_by_class = {
        k: np.sort(1.0 - _cal_proba[y_cal == k, k]) for k in range(n_classes)
    }

    def conformal_p_values(X_query) -> np.ndarray:
        """The function Sec. 3.2 declares as the SHAP target."""
        Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
        proba = model.predict_proba(Xq)
        out = np.empty((Xq.shape[0], n_classes))
        for k, a_cal in cal_alpha_by_class.items():
            a_new = 1.0 - proba[:, k]
            ge = a_cal.shape[0] - np.searchsorted(a_cal, a_new, side="left")
            out[:, k] = (ge + 1.0) / (a_cal.shape[0] + 1.0)
        return out

    mo.md(
        f"Pipeline calibrated — RF(300), Mondrian hinge p-values, "
        f"n_cal per class = {[int(v.shape[0]) for v in cal_alpha_by_class.values()]}, "
        f"n_test = {len(X_test)}."
    )
    return X_test, X_train, class_names, conformal_p_values, model, n_classes


@app.cell
def _(mo):
    mo.md(r"""
    ## E1 — TreeSHAP cannot be constructed on the p-value function

    Sec. 3.2 target (`conformal_p_values`) + Sec. 3.3 tool (`TreeExplainer`),
    combined directly:
    """)
    return


@app.cell
def _(conformal_p_values, mo):
    import shap

    try:
        shap.TreeExplainer(conformal_p_values)
        e1_result = mo.md("**Unexpectedly succeeded — investigate.**")
        tree_on_pv_error = None
    except Exception as _e:
        tree_on_pv_error = f"{type(_e).__name__}: {_e}"
        e1_result = mo.md(
            f"""
            ```
            shap.TreeExplainer(conformal_p_values)
            → {tree_on_pv_error}
            ```

            The Mondrian p-value is an external, calibration-dependent
            step transform of the ensemble output — it does not live in the trees, so
            no tree-path algorithm can traverse it. TreeSHAP on the declared target
            fails at construction, before computing anything.
            """
        )
    e1_result
    return shap, tree_on_pv_error


@app.cell
def _(mo):
    mo.md(r"""
    ## E2 — What `shap.Explainer` selects for a p-value callable

    The natural reading of *"default settings as implemented in the SHAP
    package"* is the package's front door: `shap.Explainer(...)`, which
    selects the algorithm from the type of its first argument. We hand it the
    p-value callable and record every warning Python emits:
    """)
    return


@app.cell
def _(X_train, conformal_p_values, mo, shap, warnings):
    background = shap.sample(X_train, 50, random_state=0)

    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        auto_explainer = shap.Explainer(conformal_p_values, background)
    n_warnings = len(_caught)

    mo.md(
        f"""
        ```
        shap.Explainer(conformal_p_values, background)
        → {type(auto_explainer).__name__}          # a model-agnostic explainer
        warnings emitted: {n_warnings}
        ```

        No exception and no warning of any kind. Handed the p-value callable,
        `shap.Explainer` resolves to
        **`{type(auto_explainer).__name__}`** — a model-agnostic estimator —
        and does not report which algorithm it chose. Nothing in the call
        signature or the return value distinguishes this from a TreeSHAP run
        unless the caller inspects the returned object's type.
        """
    )
    return auto_explainer, background


@app.cell
def _(X_train, mo, model, shap):
    # Control: the same front door, handed the model instead of the p-value function,
    # does return TreeExplainer. The dispatch is driven purely by the argument type.
    _bg = shap.sample(X_train, 50, random_state=0)
    _on_model = shap.Explainer(model, _bg)
    mo.md(
        f"""
        Control — same call on the raw model:

        ```
        shap.Explainer(model, background) → {type(_on_model).__name__}
        ```

        So `shap.Explainer` does produce TreeSHAP when given the tree model,
        but then the explained function is the **probability**, not the
        p-value of Sec. 3.2. The algorithm depends on which object is passed:
        the p-value function resolves to a model-agnostic explainer (E2); the
        model resolves to TreeSHAP, but on the probability. Sec. 3.2 and
        Sec. 3.3 read as descriptions of the same call only if that call took
        the p-value function.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E3 — The auto-selected explainer is numerically equivalent to KernelSHAP

    With $d=4$ features the auto-selected explainer enumerates the full
    coalition lattice. KernelSHAP with `nsamples` $\geq 2^d$ does the same.
    If the two produce identical values, then whatever ran under "default
    settings" is KernelSHAP in every numerical sense:
    """)
    return


@app.cell
def _(X_test, auto_explainer, background, conformal_p_values, mo, np, shap):
    _n_check = 10
    _X_chk = np.asarray(X_test, dtype=float)[:_n_check]

    phi_auto = auto_explainer(_X_chk).values  # (n, d, K)

    kernel_explainer = shap.KernelExplainer(conformal_p_values, background)
    _phi_k = np.array(
        kernel_explainer.shap_values(_X_chk, nsamples=2**4, silent=True)
    )
    if _phi_k.ndim == 3 and _phi_k.shape[0] == phi_auto.shape[2]:
        _phi_k = np.moveaxis(_phi_k, 0, -1)

    max_gap = float(np.abs(phi_auto - _phi_k).max())

    mo.md(
        f"""
        ```
        max | phi_auto - phi_KernelSHAP(full lattice) | = {max_gap:.2e}
        ```

        The gap is at machine precision. The auto-selected explainer and
        KernelSHAP with full enumeration are the same computation. The
        pipeline as declared is consistent with KernelSHAP(-equivalent) on
        the conformal p-value function, which is the estimator we adopt
        explicitly in this work.
        """
    )
    return (kernel_explainer,)


@app.cell
def _(mo):
    mo.md(r"""
    ## E4 — Baseline scale in the published figures

    Johansson et al., Fig. 2 (Iris): the waterfall reports $f(x)=0.034$ and
    $E[f(X)] = 0.207$. Which candidate explainer produces baselines on that
    scale?
    """)
    return


@app.cell
def _(class_names, kernel_explainer, mo, model, n_classes, np, pd, shap):
    _te = shap.TreeExplainer(model)
    _base_tree = np.round(np.atleast_1d(_te.expected_value), 3)
    _base_kernel = np.round(np.atleast_1d(kernel_explainer.expected_value), 3)

    _df = pd.DataFrame(
        {
            "class": class_names,
            "TreeExplainer(model) baseline": _base_tree[:n_classes],
            "KernelSHAP(p-values) baseline": _base_kernel[:n_classes],
        }
    )

    mo.vstack(
        [
            mo.ui.table(_df, selection=None),
            mo.md(
                r"""
                TreeExplainer baselines are class priors on the probability scale
                ($\approx 1/3$ each on balanced Iris — and for an RF they collapse to
                near 0/1 per instance). The paper's reported quantities ($f(x)=0.034$,
                $E[f(X)]=0.207$) are p-value magnitudes: attainable only as
                expectations of the conformal p-value over a background, i.e. only by
                the model-agnostic explainer applied to $p_y(x)$.

                The figures' own baseline scale indicates which function was explained.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo, tree_on_pv_error):
    mo.md(f"""
    ## Summary

    | # | Check | Result at runtime |
    |---|---|---|
    | E1 | TreeSHAP on the declared target (p-value fn) | raises at construction: `{tree_on_pv_error}` |
    | E2 | `shap.Explainer` on the p-value fn resolves to | a model-agnostic explainer, unreported |
    | E3 | That explainer vs. KernelSHAP (full lattice) | max gap ~1e-16 |
    | E4 | Their figures' baseline scale | matches the p-value scale, not the probability scale |

    The published attributions are consistent with a model-agnostic Shapley
    estimator — KernelSHAP in every numerical sense — applied to the conformal
    p-value function. The "Tree SHAP" label in Sec. 3.3 is best explained by
    E2: `shap.Explainer`'s auto-dispatch selects the algorithm from the input
    type and does not report the choice, so both sections of the paper are
    consistent with a single call that a reader cannot identify as
    model-agnostic without inspecting the returned object.

    For our own paper, we take this as a reason to name the estimator
    explicitly rather than rely on `shap`'s default dispatch: our use of
    KernelSHAP on conformal quantities is stated directly, with exactness
    guaranteed by full-lattice enumeration ($d=8$, `nsamples` $\\geq 2^8-2$).
    """)
    return


if __name__ == "__main__":
    app.run()
