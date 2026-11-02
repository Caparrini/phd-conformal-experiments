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
    # globafa2026 — Alpha Sensitivity

    Recalibration of the Mondrian LAC conformal predictor at multiple alpha
    levels, on the classifier and calibration set already fitted for
    `globafa2026` (run `globafa2026-xgboost`). No retraining: XGBoost is
    loaded from MLflow and reused unchanged; only the calibration quantiles
    q_hat depend on alpha.
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

    Package versions, seeds, and commit hashes, logged as MLflow params.
    `conformalpy` is an editable path dependency (`pyproject.toml`), so
    `uv.lock` cannot pin it; its commit is the only record of which
    conformal implementation produced these numbers.
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
    }
    mo.md(
        "**Provenance**\n\n"
        + "\n".join(f"- `{k}` = `{v}`" for k, v in provenance.items())
    )
    return np, pd, provenance


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

    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{cfg.experiment.run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0].run_id

    cache_dir = EXPERIMENT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pipeline_path = cache_dir / f"pipeline_{run_id}.joblib"

    if cached_pipeline_path.exists():
        pipeline = joblib.load(cached_pipeline_path)
        mo.md(f"Loaded model from local cache (`{cached_pipeline_path.name}`), run `{run_id}`")
    else:
        pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        joblib.dump(pipeline, cached_pipeline_path)
        mo.md(f"Loaded model from MLflow run `{run_id}` and cached locally")
    return mlflow, pipeline


@app.cell
def _(mo):
    mo.md("""
    ## Baseline calibration and p-value matrix

    Under LAC, `p_y(x) = (N_> + N_= + 1) / (N_cal + 1)` per class is
    deterministic and exactly equivalent to the set membership produced by
    `predict()` (`ConformalClassifier.predict_p_values` docstring). One
    p-value matrix therefore determines the prediction sets for every alpha;
    no recalibration is needed to sweep alpha.
    """)
    return


@app.cell
def _(X_cal, X_test, cfg, pipeline, y_cal):
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity

    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=cfg.conformal.alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)

    P = conf_clf.predict_p_values(X_test, smoothed=False)
    return ConformalClassifier, P, conf_clf, lac_nonconformity


@app.cell
def _(mo):
    mo.md("""
    ## Equivalence check

    For every alpha in the coarse grid, the prediction sets derived from `P`
    via `{y : p_y(x) > alpha}` must equal `conf_clf.predict(X_test,
    alpha=alpha)`, which recomputes q_hat independently through
    `_compute_qhat_for_alpha` and applies it to nonconformity scores. This is
    an independent numerical cross-check of the p-value/quantile duality
    claimed by `predict_p_values`'s docstring, not a tautology: the two code
    paths share no intermediate state beyond the calibration scores.
    """)
    return


@app.cell
def _(P, X_test, cfg, conf_clf, mo):
    from dslib.alpha_sweep import sets_from_p_values

    _n_mismatches = {}
    for _alpha in cfg.alpha_sensitivity.grid_coarse:
        _sets_from_p = sets_from_p_values(P, _alpha)
        _sets_from_predict = [set(s) for s in conf_clf.predict(X_test, alpha=float(_alpha))]
        _mismatch = sum(a != b for a, b in zip(_sets_from_p, _sets_from_predict))
        _n_mismatches[_alpha] = _mismatch

    equivalence_report = _n_mismatches
    assert all(v == 0 for v in equivalence_report.values()), (
        f"Mismatches between p-value sets and predict() sets: {equivalence_report}"
    )
    mo.md(
        "Mismatches per alpha (must all be 0): "
        + ", ".join(f"α={a}: {n}" for a, n in equivalence_report.items())
    )
    return equivalence_report, sets_from_p_values


@app.cell
def _(mo):
    mo.md("""
    ## Fine-grid sweep
    """)
    return


@app.cell
def _(P, cfg, mo, np, y_test):
    from dslib.alpha_sweep import check, sweep

    _g = cfg.alpha_sensitivity.grid_fine
    alphas_fine = np.round(np.arange(_g.start, _g.stop, _g.step), 4)

    df_fine = sweep(P, y_test.values, alphas_fine)
    check(df_fine, P, y_test.values)

    mo.md(f"Sweep computed for {len(alphas_fine)} alpha values; `check()` passed.")
    return alphas_fine, df_fine


@app.cell
def _(df_fine, mo):
    mo.ui.table(df_fine)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Coarse-grid table (paper, section 5.8)

    The α=0.10 row must reproduce, digit for digit, the metrics already
    logged under run `globafa2026-conformal-evaluation`:
    `test_coverage=0.9132460106499125`,
    `test_coverage_class_0=0.9254879259014225`,
    `test_coverage_class_1=0.8711993079674125`,
    `test_avg_set_size=1.6138256710056278`,
    `test_singleton_rate=0.38617432899437226`,
    `test_accuracy_when_confident=0.7753501907389816`, `test_empty_rate=0.0`.
    """)
    return


@app.cell
def _(cfg, df_fine, mo):
    from dslib.alpha_sweep import table_for_paper, to_latex

    table_body = table_for_paper(df_fine, list(cfg.alpha_sensitivity.grid_coarse))

    _row_010 = table_body[table_body["alpha"] == 0.10].iloc[0]
    _expected = {
        "coverage": 0.9132460106499125,
        "coverage_class_0": 0.9254879259014225,
        "coverage_class_1": 0.8711993079674125,
        "avg_set_size": 1.6138256710056278,
        "singleton_rate": 0.38617432899437226,
        "accuracy_given_confident": 0.7753501907389816,
    }
    for _col, _exp in _expected.items():
        assert abs(_row_010[_col] - _exp) < 1e-9, (
            f"{_col} at alpha=0.10 is {_row_010[_col]}, expected {_exp} "
            "(regression against globafa2026-conformal-evaluation)."
        )
    assert _row_010["empty_rate"] == 0.0, f"empty_rate at alpha=0.10 is {_row_010['empty_rate']}, expected 0.0"

    mo.ui.table(table_body)
    return table_body, to_latex


@app.cell
def _(mo, table_body, to_latex):
    mo.md(f"""
    ```latex\n{to_latex(table_body)}\n```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Marginal precision of newly-formed singletons
    """)
    return


@app.cell
def _(P, alphas_fine, mo, y_test):
    from dslib.alpha_sweep import marginal_precision

    df_marginal = marginal_precision(P, y_test.values, alphas_fine)
    mo.ui.table(df_marginal)
    return (df_marginal,)


@app.cell
def _(mo):
    mo.md("""
    ## Transition matrices between coarse alpha levels
    """)
    return


@app.cell
def _(P, cfg, mo, y_test):
    from dslib.alpha_sweep import transitions

    transition_tables = {
        (a1, a2): transitions(P, y_test.values, a1, a2)
        for a1, a2 in cfg.alpha_sensitivity.transitions
    }
    for (_a1, _a2), _tab in transition_tables.items():
        mo.output.append(mo.md(f"**α={_a1} → α={_a2}**"))
        mo.output.append(mo.ui.table(_tab.reset_index(names="state_at_" + str(_a1))))
    return (transition_tables,)


@app.cell
def _(mo):
    mo.md("""
    ## Empty-set impossibility

    Under binary LAC, `s(x,0) + s(x,1) = 1` for every x, so an empty set
    requires `q0 + q1 < 1`. `q_hat` per class is recomputed at each alpha by
    recalibrating a fresh `ConformalClassifier` on the same `(pipeline,
    X_cal, y_cal)` — the model is not retrained, only the calibration
    quantile changes.
    """)
    return


@app.cell
def _(
    ConformalClassifier,
    X_cal,
    alphas_fine,
    lac_nonconformity,
    mo,
    np,
    pd,
    pipeline,
    y_cal,
):
    from dslib.alpha_sweep import empty_set_threshold

    _rows = []
    for _alpha in alphas_fine:
        _clf = ConformalClassifier(
            model=pipeline, alpha=float(_alpha),
            nonconformity_function=lac_nonconformity, mondrian=True,
        )
        _clf.calibrate(X_cal, y_cal)
        _q = _clf.qhat_per_class_
        _est = empty_set_threshold(_q[0], _q[1])
        _rows.append(
            {
                "alpha": float(_alpha),
                "q0": _q[0],
                "q1": _q[1],
                "q0_minus_q1": _q[0] - _q[1],
                "sum_q": _est.sum_q,
                "empty_possible": _est.empty_possible,
            }
        )
    df_qhat = pd.DataFrame(_rows)

    _sign = np.sign(df_qhat["q0_minus_q1"])
    _sign_changes = df_qhat.loc[_sign.diff().fillna(0) != 0, "alpha"].tolist()
    qhat_asymmetry_sign_constant = len(_sign_changes) == 0

    _crossover = df_qhat[df_qhat["empty_possible"]]
    empty_crossover_alpha = float(_crossover["alpha"].min()) if len(_crossover) > 0 else None

    if qhat_asymmetry_sign_constant:
        _sign_report = (
            f"q0 - q1 keeps sign `{'+' if _sign.iloc[0] > 0 else '-'}` across "
            f"α ∈ [{alphas_fine.min()}, {alphas_fine.max()}]."
        )
    else:
        _flip_rows = df_qhat[df_qhat["alpha"].isin(_sign_changes)]
        _detail = ", ".join(
            f"α={r.alpha:.2f}: q0-q1={r.q0_minus_q1:+.6f}" for r in _flip_rows.itertuples()
        )
        _sign_report = (
            f"q0 - q1 changes sign at α ∈ {_sign_changes} "
            f"({_detail}). At the coarse grid used in the body table, this means "
            f"q0 - q1 is negative at α=0.05 and positive at α ∈ {{0.10, 0.15, 0.20}}: "
            "the asymmetry does not hold across the full range requested."
        )

    _empty_report = (
        f"Empty sets become possible at α ≥ {empty_crossover_alpha}."
        if empty_crossover_alpha is not None
        else "Empty sets remain impossible (q0 + q1 > 1) for every α tested."
    )

    mo.md(f"{_sign_report}\n\n{_empty_report}")
    return df_qhat, empty_crossover_alpha, qhat_asymmetry_sign_constant


@app.cell
def _(df_qhat, mo):
    mo.ui.table(df_qhat)
    return


@app.cell
def _(mo):
    mo.md("""
    ## FCOD gradient direction across alpha

    Same smoothing parameters and `dti_n` clip as `04_fcod.py`
    (`n_grid`, `percentile_range`, `sigma` from `cfg.alpha_sensitivity.fcod`).
    Prediction sets for each alpha come from `sets_from_p_values(P, alpha)`
    — the same p-value matrix used throughout this notebook, not a
    recomputation from disk.
    """)
    return


@app.cell
def _(P, X_test, cfg, mo, np, pd, sets_from_p_values, y_test):
    from conformalpy.fcod import compute_fcod_smoothed, merge_uncertain_outcomes

    _fcod_cfg = cfg.alpha_sensitivity.fcod
    _clip = {
        "fico_n": None,
        "dti_n": (0, float(X_test["dti_n"].quantile(0.99))),
    }
    y_test_arr = np.asarray(y_test)

    _rows = []
    for _feat in cfg.alpha_sensitivity.fcod_features:
        _values = X_test[_feat].values
        if _clip.get(_feat) is not None:
            _lo, _hi = _clip[_feat]
            _mask = (_values >= _lo) & (_values <= _hi)
        else:
            _mask = np.ones(len(_values), dtype=bool)
        _values_viz = _values[_mask]
        _y_viz = y_test_arr[_mask]

        for _alpha in cfg.alpha_sensitivity.grid_coarse:
            _sets = sets_from_p_values(P, _alpha)
            _sets_viz = [s for s, m in zip(_sets, _mask) if m]

            _fcod = compute_fcod_smoothed(
                _values_viz, _sets_viz, _y_viz,
                n_grid=_fcod_cfg.n_grid,
                percentile_range=tuple(_fcod_cfg.percentile_range),
                sigma=_fcod_cfg.sigma,
            )
            _merged = merge_uncertain_outcomes(_fcod)
            _grid = _merged["grid"]
            _ts_slope = float(np.polyfit(_grid, _merged["TS"], 1)[0])
            _sc_slope = float(np.polyfit(_grid, _merged["SC"], 1)[0])
            _rows.append(
                {"feature": _feat, "alpha": float(_alpha), "TS_slope": _ts_slope, "SC_slope": _sc_slope}
            )

    df_fcod_gradients = pd.DataFrame(_rows)

    _sign_flip_report = []
    fcod_gradient_signs_constant = True
    for _feat in cfg.alpha_sensitivity.fcod_features:
        _sub = df_fcod_gradients[df_fcod_gradients["feature"] == _feat]
        for _col in ["TS_slope", "SC_slope"]:
            _sign = np.sign(_sub[_col])
            if not (_sign == _sign.iloc[0]).all():
                fcod_gradient_signs_constant = False
                _sign_flip_report.append(
                    f"{_feat}/{_col}: " + ", ".join(
                        f"α={r.alpha:.2f}: {r_val:+.6g}"
                        for r, r_val in zip(_sub.itertuples(), _sub[_col])
                    )
                )

    if fcod_gradient_signs_constant:
        mo.output.append(mo.md(
            "FCOD gradient direction (TS and SC slope) is constant in sign "
            f"across α ∈ {list(cfg.alpha_sensitivity.grid_coarse)} for "
            f"{list(cfg.alpha_sensitivity.fcod_features)}."
        ))
    else:
        mo.output.append(mo.md(
            "FCOD gradient direction changes sign across the coarse alpha grid "
            "for at least one feature/outcome pair:\n\n"
            + "\n\n".join(_sign_flip_report)
        ))

    mo.output.append(mo.ui.table(df_fcod_gradients))
    return df_fcod_gradients, fcod_gradient_signs_constant


@app.cell
def _(mo):
    mo.md("""
    ## Evolution and risk-coverage figures
    """)
    return


@app.cell
def _(df_fine, y_test):
    from dslib.alpha_sweep import plot_evolution, plot_risk_coverage

    fig_evolution, _ = plot_evolution(df_fine, base_rate=1 - float(y_test.mean()))
    fig_evolution
    return fig_evolution, plot_risk_coverage


@app.cell
def _(df_fine, plot_risk_coverage):
    fig_risk_coverage, _ = plot_risk_coverage(df_fine)
    fig_risk_coverage
    return (fig_risk_coverage,)


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
    df_fcod_gradients,
    df_fine,
    df_marginal,
    df_qhat,
    empty_crossover_alpha,
    equivalence_report,
    fcod_gradient_signs_constant,
    fig_evolution,
    fig_risk_coverage,
    mlflow,
    mlflow_config,
    mo,
    provenance,
    qhat_asymmetry_sign_constant,
    table_body,
    to_latex,
    transition_tables,
):
    from dslib.tracking import tracked_run
    import json
    import tempfile
    import os

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_fine.to_csv(results_dir / "alpha_sweep_fine.csv", index=False)
    table_body.to_csv(results_dir / "alpha_sweep_table.csv", index=False)
    with open(results_dir / "alpha_sweep_table.tex", "w") as _f:
        _f.write(to_latex(table_body))
    df_marginal.to_csv(results_dir / "marginal_precision.csv", index=False)
    df_qhat.to_csv(results_dir / "qhat_by_alpha.csv", index=False)
    df_fcod_gradients.to_csv(results_dir / "fcod_gradients.csv", index=False)
    for (_a1, _a2), _tab in transition_tables.items():
        _tab.to_csv(results_dir / f"transitions_{_a1}_to_{_a2}.csv")

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(
        config_dict, data_path, cfg.experiment.analysis_experiment_name,
        run_name="globafa2026-alpha-sensitivity",
        mlflow_config=mlflow_config,
    ):
        for _k, _v in provenance.items():
            mlflow.set_tag(f"provenance.{_k}", _v)

        mlflow.log_artifact(str(results_dir / "alpha_sweep_fine.csv"), artifact_path="alpha_sweep")
        mlflow.log_artifact(str(results_dir / "alpha_sweep_table.csv"), artifact_path="alpha_sweep")
        mlflow.log_artifact(str(results_dir / "alpha_sweep_table.tex"), artifact_path="alpha_sweep")
        mlflow.log_artifact(str(results_dir / "marginal_precision.csv"), artifact_path="alpha_sweep")
        mlflow.log_artifact(str(results_dir / "qhat_by_alpha.csv"), artifact_path="alpha_sweep")
        mlflow.log_artifact(str(results_dir / "fcod_gradients.csv"), artifact_path="alpha_sweep")
        for _a1, _a2 in cfg.alpha_sensitivity.transitions:
            mlflow.log_artifact(
                str(results_dir / f"transitions_{_a1}_to_{_a2}.csv"), artifact_path="alpha_sweep"
            )

        mlflow.log_figure(fig_evolution, "alpha_sweep/alpha_evolution.png", save_kwargs={"bbox_inches": "tight"})
        mlflow.log_figure(fig_risk_coverage, "alpha_sweep/risk_coverage.png", save_kwargs={"bbox_inches": "tight"})

        with tempfile.TemporaryDirectory() as _tmp:
            _p = os.path.join(_tmp, "equivalence_check.json")
            with open(_p, "w") as _f:
                json.dump(equivalence_report, _f)
            mlflow.log_artifact(_p, artifact_path="alpha_sweep")

        # check() and the p-value/predict() equivalence assertion ran earlier
        # in this notebook and would have stopped execution on failure; only
        # the two directional robustness properties are recorded here as
        # findings rather than hard assertions, since they concern the data
        # rather than the module's mechanics (see cells above).
        mlflow.set_tag("alpha_sweep_checks", "passed")
        mlflow.set_tag("qhat_asymmetry_sign_constant", qhat_asymmetry_sign_constant)
        mlflow.set_tag("fcod_gradient_signs_constant", fcod_gradient_signs_constant)
        if empty_crossover_alpha is not None:
            mlflow.log_metric("empty_crossover_alpha", empty_crossover_alpha)

    mo.md(
        f"Logged to MLflow run `globafa2026-alpha-sensitivity` under experiment "
        f"`{cfg.experiment.analysis_experiment_name}` (not `{cfg.experiment.name}` — "
        "this notebook never logs a model, so it stays out of the experiment "
        "whose most-recent-run is expected to have one)."
    )
    return


if __name__ == "__main__":
    app.run()
