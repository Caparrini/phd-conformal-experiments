import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="globafa2026")

    mo.md("""
    # globafa2026 — Exact Shapley vs KernelSHAP

    Exact Shapley values by full coalition enumeration (M=8 features, 256
    coalitions), computed on the same 1000 test instances already explained
    with KernelSHAP in `05_shap.py`, using the identical value function and
    background/explain sampling — reused, not rewritten.
    """)
    return EXPERIMENT_DIR, OmegaConf, cfg, mo


@app.cell
def _():
    """Initialize centralized MLflow configuration."""
    from dslib.mlflow_config import load_mlflow_config

    mlflow_config = load_mlflow_config()
    mlflow_config.validate_and_log()
    return (mlflow_config,)


@app.cell
def _(mo):
    mo.md("""
    ## Provenance

    Package versions, seeds, and commit hashes. `shap.KernelExplainer` clamps
    `nsamples` to `2**M - 2` whenever `M <= 30`
    (`shap/explainers/_kernel.py:407-411`); for `M=8` that is 254, not the
    1024 configured in `cfg.shap.nsamples` — the effective sample count is
    logged explicitly below.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, cfg, mo):
    import subprocess
    import xgboost
    import shap
    import sklearn
    import numpy as np
    import pandas as pd
    import scipy
    import mlflow as _mlflow_pkg

    def _git_commit(path) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    _repo_root = EXPERIMENT_DIR.parent.parent
    _conformalpy_dir = _repo_root.parent / "conformalpy"

    M_features = len(cfg.data.features.numerical) + len(cfg.data.features.categorical)
    nsamples_nominal = int(cfg.shap.nsamples)
    nsamples_effective = 2**M_features - 2 if M_features <= 30 else nsamples_nominal

    provenance = {
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "marimo": mo.__version__ if hasattr(mo, "__version__") else "0.20.2",
        "mlflow": _mlflow_pkg.__version__,
        "conformalpy_commit": _git_commit(_conformalpy_dir),
        "repo_commit": _git_commit(_repo_root),
        "experiment.seed": cfg.experiment.seed,
        "model.random_state": cfg.model.params.random_state,
        "shap.n_background": int(cfg.shap.n_background),
        "shap.n_explain": int(cfg.shap.n_explain),
        "M_features": M_features,
        "nsamples_nominal": nsamples_nominal,
        "nsamples_effective": nsamples_effective,
    }
    mo.md(
        "**Provenance**\n\n"
        + "\n".join(f"- `{k}` = `{v}`" for k, v in provenance.items())
    )
    return M_features, np, pd, provenance


@app.cell
def _(EXPERIMENT_DIR, cfg):
    from dslib.data_utils import load_and_split

    data_path = EXPERIMENT_DIR / cfg.data.file
    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    splits = load_and_split(
        data_path=data_path,
        date_column=cfg.data.splits.date_column,
        train_cutoff=cfg.data.splits.train_cutoff,
        calibration_cutoff=cfg.data.splits.calibration_cutoff,
        target=cfg.data.target,
        features=feature_cols,
    )

    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    return X_cal, X_test, data_path, y_cal, y_test


@app.cell
def _(EXPERIMENT_DIR, cfg, mo):
    import mlflow
    import joblib
    import re

    cache_dir = EXPERIMENT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # A pipeline cached by 06_alpha_sensitivity.py (same naming convention)
    # lets this notebook run without depending on MLflow being reachable;
    # the model itself is still the one trained under globafa2026-xgboost,
    # not retrained.
    _cached = list(cache_dir.glob("pipeline_*.joblib"))
    if _cached:
        cached_pipeline_path = _cached[0]
        run_id = re.sub(r"^pipeline_|\.joblib$", "", cached_pipeline_path.name)
        pipeline = joblib.load(cached_pipeline_path)
        mo.md(f"Loaded model from local cache (`{cached_pipeline_path.name}`), run `{run_id}`")
    else:
        experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{cfg.experiment.run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        run_id = runs.iloc[0].run_id
        cached_pipeline_path = cache_dir / f"pipeline_{run_id}.joblib"
        pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        joblib.dump(pipeline, cached_pipeline_path)
        mo.md(f"Loaded model from MLflow run `{run_id}` and cached locally")
    return cache_dir, mlflow, pipeline, run_id


@app.cell
def _(X_cal, cfg, pipeline, y_cal):
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity

    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=cfg.conformal.alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)
    return (conf_clf,)


@app.cell
def _(mo):
    mo.md("""
    ## Sampling design

    Categorical features are integer-encoded into one column each; the
    pipeline's `OneHotEncoder` expands them internally after decoding. This
    keeps the coalition space at 8 semantic features (256 coalitions), not
    one per one-hot dummy — no coalition-count explosion. Categoricals are
    not fixed to their mode: all 8 features vary, matching `05_shap.py`.
    """)
    return


@app.cell
def _(M_features, X_test, cfg, mo, y_test):
    from dslib.conformal_shap import build_shap_design

    numeric_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)
    assert len(numeric_cols) + len(categorical_cols) == M_features == 8

    design = build_shap_design(
        X_test, y_test, numeric_cols, categorical_cols,
        seed=cfg.experiment.seed,
        n_background=int(cfg.shap.n_background),
        n_explain=int(cfg.exact_shapley.n_exact),
    )
    mo.md(
        f"Background: {design.X_background.shape[0]} rows. "
        f"Explain: {design.X_explain.shape[0]} rows, {design.X_explain.shape[1]} features "
        f"→ {2**design.X_explain.shape[1]} coalitions total, "
        f"{2**design.X_explain.shape[1] - 2} non-trivial."
    )
    return categorical_cols, design, numeric_cols


@app.cell
def _(mo):
    mo.md("""
    ## Equivalence check against `05_shap.py`'s closure

    The value function used here (`IntegerEncodedPValueFunction`) must
    reproduce the `predict_p_values_fn` closure of
    `experiments/globafa2026/05_shap.py:162-169` exactly — reconstructed
    verbatim below and compared row for row, not assumed.
    """)
    return


@app.cell
def _(categorical_cols, conf_clf, design, mo, np, numeric_cols, pd):
    from dslib.conformal_shap import IntegerEncodedPValueFunction

    _all_cols = numeric_cols + categorical_cols
    _n_num = len(numeric_cols)
    _cat_categories = design.cat_categories
    _cat_to_int = {col: {v: i for i, v in enumerate(cats)} for col, cats in _cat_categories.items()}
    _int_to_cat = {col: {i: v for v, i in m.items()} for col, m in _cat_to_int.items()}

    def _reference_predict_p_values_fn(X: np.ndarray) -> np.ndarray:
        """Verbatim copy of experiments/globafa2026/05_shap.py:162-169."""
        data = {col: X[:, i].astype(float) for i, col in enumerate(numeric_cols)}
        for j, col in enumerate(categorical_cols):
            n_cats = len(_cat_categories[col])
            codes = np.round(X[:, _n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [_int_to_cat[col][c] for c in codes]
        return conf_clf.predict_p_values(pd.DataFrame(data, columns=_all_cols))

    predict_p_values_fn = IntegerEncodedPValueFunction(
        conf_clf, numeric_cols, categorical_cols, design.cat_categories
    )

    _out_ref = _reference_predict_p_values_fn(design.X_explain)
    _out_new = predict_p_values_fn(design.X_explain)
    assert np.array_equal(_out_ref, _out_new), (
        "IntegerEncodedPValueFunction diverges from the 05_shap.py closure "
        f"on {int(np.sum(np.any(_out_ref != _out_new, axis=1)))} of "
        f"{len(_out_ref)} explained instances."
    )

    mo.md(f"Value function verified equivalent on all {len(design.X_explain)} explained instances.")
    return (predict_p_values_fn,)


@app.cell
def _(mo):
    mo.md("""
    ## KernelSHAP (same cache key as `05_shap.py`)
    """)
    return


@app.cell
def _(cache_dir, cfg, design, mo, predict_p_values_fn, run_id):
    from dslib.shap_cache import compute_or_load_shap

    shap_kernel = compute_or_load_shap(
        predict_p_values_fn, design.X_background, design.X_explain,
        nsamples=int(cfg.shap.nsamples),
        model_id=f"{run_id}-conformal",
        cache_dir=cache_dir,
    )
    mo.md(f"KernelSHAP: `{shap_kernel.shape}`")
    return (shap_kernel,)


@app.cell
def _(mo):
    mo.md("""
    ## Exact Shapley (full 2^8 coalition enumeration)
    """)
    return


@app.cell
def _(design, mo, predict_p_values_fn):
    from dslib.conformal_shap import exact_shapley

    shap_exact = exact_shapley(predict_p_values_fn, design.X_background, design.X_explain)
    mo.md(f"Exact Shapley: `{shap_exact.shape}`")
    return (shap_exact,)


@app.cell
def _(mo):
    mo.md("""
    ## Margins
    """)
    return


@app.cell
def _(design, shap_exact, shap_kernel):
    from conformalpy.explainability import derive_margin_shap, select_class_shap

    def _margins(shap_values, y_explain):
        p0 = shap_values[:, :, 0]
        p1 = shap_values[:, :, 1]
        evidence = p1 - p0
        p_true = select_class_shap(shap_values, y_explain)
        p_false = select_class_shap(shap_values, 1 - y_explain)
        correctness = derive_margin_shap(p_true, p_false)
        return evidence, correctness

    kernel_evidence, kernel_correctness = _margins(shap_kernel, design.y_explain)
    exact_evidence, exact_correctness = _margins(shap_exact, design.y_explain)
    return (
        exact_correctness,
        exact_evidence,
        kernel_correctness,
        kernel_evidence,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Verification: exact enumeration reproduces KernelSHAP
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Under exact enumeration, KernelSHAP and `exact_shapley` solve the same
    combinatorial average over the same 254 non-trivial coalitions and the
    same background, so they agree to floating-point precision on almost
    every instance. The conformal p-value is a step function of the
    nonconformity score (`p_y(x) = (N_> + N_= + 1)/(N_cal+1)`,
    `conformal_classifier.py:834-921`), quantized in steps of
    `1/(N_cal_y+1)`; the two implementations construct their masked design
    matrices in different array operations, so on rare (instance, coalition)
    pairs that sit exactly on a nonconformity-score tie, floating-point
    order can push one implementation to one side of the tie and the other
    to the other side, producing a jump of at most one quantization step for
    that pair. This is bounded and reported below per instance, not asserted
    away — the tolerance check below only guards against a discrepancy far
    larger than this bound, which would indicate an actual implementation
    bug rather than a tie.
    """)
    return


@app.cell
def _(
    exact_correctness,
    exact_evidence,
    kernel_correctness,
    kernel_evidence,
    mo,
    np,
    y_cal,
):
    from scipy.stats import spearmanr

    _quantum = 1.0 / (min(int((y_cal == 0).sum()), int((y_cal == 1).sum())) + 1)
    # A discrepancy this large cannot come from a single tie-boundary flip
    # on the finest-grained class and would indicate a real divergence
    # between the two estimators.
    _bug_tolerance = 50 * _quantum

    def _agreement(kernel, exact, name):
        diff = np.abs(kernel - exact)
        max_abs_diff = float(diff.max())
        per_instance_max = diff.max(axis=1)
        n_tied = int(np.sum(per_instance_max > 1e-9))
        rho, _ = spearmanr(
            np.abs(kernel).mean(axis=0), np.abs(exact).mean(axis=0)
        )
        return {
            "margin": name,
            "max_abs_diff": max_abs_diff,
            "n_instances_with_tie_boundary_diff": n_tied,
            "n_instances_total": len(per_instance_max),
            "rank_correlation": float(rho),
        }

    verification = [
        _agreement(kernel_evidence, exact_evidence, "evidence"),
        _agreement(kernel_correctness, exact_correctness, "correctness"),
    ]
    for _v in verification:
        assert _v["max_abs_diff"] < _bug_tolerance, (
            f"KernelSHAP and exact Shapley diverge on the {_v['margin']} margin "
            f"by {_v['max_abs_diff']}, which exceeds {_bug_tolerance} "
            f"(50x the p-value quantization step {_quantum:.3e}); this is too "
            "large to be a tie-boundary effect and indicates a bug."
        )

    mo.md(
        f"p-value quantization step (finer class): {_quantum:.3e}\n\n"
        + "\n\n".join(
            f"**{v['margin']} margin**: max |Δ| = {v['max_abs_diff']:.3e} "
            f"({v['max_abs_diff']/_quantum:.1f}x quantum), "
            f"{v['n_instances_with_tie_boundary_diff']}/{v['n_instances_total']} instances "
            "show any non-machine-precision difference, "
            f"rank correlation ρ = {v['rank_correlation']:.6f}"
            for v in verification
        )
    )
    return spearmanr, verification


@app.cell
def _(mo):
    mo.md("""
    ## Dispersion table (variable × margin)
    """)
    return


@app.cell
def _(
    categorical_cols,
    exact_correctness,
    exact_evidence,
    kernel_correctness,
    kernel_evidence,
    mo,
    np,
    numeric_cols,
    pd,
    spearmanr,
):
    all_cols = numeric_cols + categorical_cols

    def _dispersion_table(kernel, exact, margin_name):
        rows = []
        mean_kernel = np.abs(kernel).mean(axis=0)
        mean_exact = np.abs(exact).mean(axis=0)
        std_kernel = np.abs(kernel).std(axis=0)
        std_exact = np.abs(exact).std(axis=0)
        q75 = np.percentile(np.abs(exact), 75, axis=0)
        q25 = np.percentile(np.abs(exact), 25, axis=0)
        rank_kernel = pd.Series(mean_kernel).rank(ascending=False).values
        rank_exact = pd.Series(mean_exact).rank(ascending=False).values
        rho, _ = spearmanr(mean_kernel, mean_exact)
        for i, col in enumerate(all_cols):
            rows.append(
                {
                    "margin": margin_name,
                    "variable": col,
                    "mean_abs_shap_kernel": mean_kernel[i],
                    "mean_abs_shap_exact": mean_exact[i],
                    "abs_diff": abs(mean_kernel[i] - mean_exact[i]),
                    "std_exact": std_exact[i],
                    "std_kernel": std_kernel[i],
                    "iqr_exact": q75[i] - q25[i],
                    "rank_kernel": rank_kernel[i],
                    "rank_exact": rank_exact[i],
                    "rank_correlation_global": rho,
                }
            )
        return pd.DataFrame(rows)

    df_dispersion = pd.concat(
        [
            _dispersion_table(kernel_evidence, exact_evidence, "evidence"),
            _dispersion_table(kernel_correctness, exact_correctness, "correctness"),
        ],
        ignore_index=True,
    )
    mo.ui.table(df_dispersion.sort_values(["margin", "mean_abs_shap_exact"], ascending=[True, False]))
    return all_cols, df_dispersion


@app.cell
def _(mo):
    mo.md("""
    ## Decision: `loan_amnt` vs `fico_n` on the evidence margin

    95% bootstrap confidence interval (2000 resamples by instance) on the
    difference of mean |SHAP| between `loan_amnt` and `fico_n`, on the
    evidence margin, using the exact Shapley values. If the interval
    contains 0, section 5.7 states a tie, as already indicated in the
    template. If it excludes 0, the leading variable can be stated as such
    — this is the finding that decides which sentence gets written, not the
    reverse.
    """)
    return


@app.cell
def _(all_cols, exact_evidence, mo, np):
    _idx_loan = all_cols.index("loan_amnt")
    _idx_fico = all_cols.index("fico_n")

    _diff_per_instance = np.abs(exact_evidence[:, _idx_loan]) - np.abs(exact_evidence[:, _idx_fico])
    _point_estimate = float(_diff_per_instance.mean())

    _rng = np.random.default_rng(0)
    _n = len(_diff_per_instance)
    _n_boot = 2000
    _boot_means = np.empty(_n_boot)
    for _b in range(_n_boot):
        _sample_idx = _rng.integers(0, _n, size=_n)
        _boot_means[_b] = _diff_per_instance[_sample_idx].mean()
    _ci_lo, _ci_hi = np.percentile(_boot_means, [2.5, 97.5])

    tie_confirmed = _ci_lo <= 0 <= _ci_hi
    decision = {
        "point_estimate": _point_estimate,
        "ci_95_lo": float(_ci_lo),
        "ci_95_hi": float(_ci_hi),
        "tie_confirmed": bool(tie_confirmed),
        "n_bootstrap": _n_boot,
        "bootstrap_seed": 0,
    }

    _verdict = (
        "The 95% CI contains 0: loan_amnt and fico_n are not distinguishable "
        "on the evidence margin under this test — section 5.7 states a tie."
        if tie_confirmed
        else "The 95% CI excludes 0: the difference in |SHAP| between loan_amnt "
        "and fico_n on the evidence margin is a real effect, not sampling "
        "noise — section 5.7 can state real leadership. This changes the "
        "argument structure of 5.7 and the Discussion and must be flagged "
        "before editing the paper text."
    )

    mo.md(
        f"mean(|SHAP_loan_amnt|) − mean(|SHAP_fico_n|) = {_point_estimate:.6f}, "
        f"95% CI = [{_ci_lo:.6f}, {_ci_hi:.6f}]\n\n{_verdict}"
    )
    return (decision,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary table for the paper (section 5.9)
    """)
    return


@app.cell
def _(df_dispersion, mo):
    summary_table = df_dispersion[
        ["margin", "variable", "mean_abs_shap_kernel", "mean_abs_shap_exact", "std_exact", "rank_correlation_global"]
    ].copy()
    mo.ui.table(summary_table)
    return (summary_table,)


@app.cell
def _(mo):
    mo.md("""
    ## Logging to MLflow
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR,
    OmegaConf,
    cfg,
    data_path,
    decision,
    df_dispersion,
    mlflow,
    mlflow_config,
    mo,
    provenance,
    summary_table,
    verification,
):
    from dslib.tracking import tracked_run

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_dispersion.to_csv(results_dir / "exact_vs_kernel_shap.csv", index=False)
    summary_table.to_csv(results_dir / "shap_dispersion.csv", index=False)

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(
        config_dict, data_path, cfg.experiment.analysis_experiment_name,
        run_name="globafa2026-exact-shapley",
        mlflow_config=mlflow_config,
    ):
        for _k, _v in provenance.items():
            mlflow.set_tag(f"provenance.{_k}", _v)

        mlflow.log_artifact(str(results_dir / "exact_vs_kernel_shap.csv"), artifact_path="exact_shapley")
        mlflow.log_artifact(str(results_dir / "shap_dispersion.csv"), artifact_path="exact_shapley")

        for _v in verification:
            mlflow.log_metric(f"max_abs_diff_{_v['margin']}", _v["max_abs_diff"])
            mlflow.log_metric(f"rank_correlation_{_v['margin']}", _v["rank_correlation"])

        mlflow.log_metric("loan_amnt_vs_fico_n_point_estimate", decision["point_estimate"])
        mlflow.log_metric("loan_amnt_vs_fico_n_ci_95_lo", decision["ci_95_lo"])
        mlflow.log_metric("loan_amnt_vs_fico_n_ci_95_hi", decision["ci_95_hi"])
        mlflow.set_tag("loan_amnt_vs_fico_n_tie_confirmed", decision["tie_confirmed"])

    mo.md(
        f"Logged to MLflow run `globafa2026-exact-shapley` under experiment "
        f"`{cfg.experiment.analysis_experiment_name}` (not `{cfg.experiment.name}` — "
        "this notebook never logs a model, so it stays out of the experiment "
        "whose most-recent-run is expected to have one)."
    )
    return


if __name__ == "__main__":
    app.run()
