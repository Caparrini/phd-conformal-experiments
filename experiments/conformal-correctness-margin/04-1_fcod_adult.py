import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import mlflow
    import json
    import math

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    from conformalpy.fcod import (
        compute_fcod_smoothed,
        compute_fcod_with_ci,
        plot_fcod,
        plot_stacked_fcod,
        plot_multi_feature_fcod,
        plot_uncertainty_zones,
    )

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="adult")

    alpha = cfg.conformal.alpha
    return (
        EXPERIMENT_DIR,
        OmegaConf,
        alpha,
        cfg,
        compute_fcod_smoothed,
        compute_fcod_with_ci,
        json,
        math,
        mlflow,
        mo,
        np,
        plot_fcod,
        plot_multi_feature_fcod,
        plot_stacked_fcod,
        plot_uncertainty_zones,
        plt,
    )


@app.cell
def _(EXPERIMENT_DIR, cfg, mo):
    from dslib.data_utils import stratified_split

    data_path = EXPERIMENT_DIR / cfg.data.file
    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    splits = stratified_split(
        data_path=data_path,
        target=cfg.data.target,
        features=feature_cols,
        train_size=cfg.data.splits.train_size,
        calibration_size=cfg.data.splits.calibration_size,
        seed=cfg.data.splits.seed,
    )

    X_test, y_test = splits["test"]

    mo.md(f"Test set: {len(X_test):,} samples")
    return X_test, data_path, y_test


@app.cell
def _(cfg, json, mlflow, mo):
    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'adult-conformal-evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0].run_id

    artifact_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/outputs/prediction_sets.json"
    )
    with open(artifact_path) as f:
        prediction_sets = json.load(f)

    mo.md(f"Loaded {len(prediction_sets):,} prediction sets from run `{run_id[:8]}`")
    return (prediction_sets,)


@app.cell
def _(
    X_test,
    cfg,
    compute_fcod_smoothed,
    compute_fcod_with_ci,
    mo,
    np,
    prediction_sets,
    y_test,
):
    fcod_features = {
        "age":            {"label": "Age",               "clip": None},
        "education-num":  {"label": "Education (years)",  "clip": None},
        "capital-gain":   {"label": "Capital Gain ($)",   "clip": (0, float(X_test["capital-gain"].quantile(0.99)))},
        "capital-loss":   {"label": "Capital Loss ($)",   "clip": (0, float(X_test["capital-loss"].quantile(0.99)))},
        "hours-per-week": {"label": "Hours per Week",     "clip": None},
    }
    N_COLS = 3

    fcod_results = {}
    fcod_results_ci = {}

    y_test_arr = np.asarray(y_test)

    for feature_name, config in fcod_features.items():
        feature_values = X_test[feature_name].values

        if config["clip"] is not None:
            lo, hi = config["clip"]
            mask = (feature_values >= lo) & (feature_values <= hi)
            feature_values_viz = feature_values[mask]
            pred_sets_viz = [prediction_sets[i] for i in range(len(prediction_sets)) if mask[i]]
            y_test_viz = y_test_arr[mask]
        else:
            feature_values_viz = feature_values
            pred_sets_viz = prediction_sets
            y_test_viz = y_test_arr

        fcod = compute_fcod_smoothed(
            feature_values_viz, pred_sets_viz, y_test_viz,
            n_grid=50, percentile_range=(5, 95),
        )
        fcod["feature_name"] = config["label"]
        fcod_results[feature_name] = fcod

        fcod_ci = compute_fcod_with_ci(
            feature_values_viz, pred_sets_viz, y_test_viz,
            n_bootstrap=100, n_grid=30, confidence_level=0.95,
            random_state=cfg.experiment.seed,
        )
        fcod_ci["feature_name"] = config["label"]
        fcod_results_ci[feature_name] = fcod_ci

    mo.md(f"Computed FCODs for {len(fcod_features)} features")
    return N_COLS, fcod_features, fcod_results, fcod_results_ci


@app.cell
def _(N_COLS, alpha, fcod_results_ci, math, plot_fcod, plt):
    n_features = len(fcod_results_ci)
    n_rows = math.ceil(n_features / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
    for i in range(n_features, n_rows * N_COLS):
        axes.flatten()[i].set_visible(False)

    for idx, (i_feature_name, i_fcod) in enumerate(fcod_results_ci.items()):
        ax = axes.flatten()[idx]
        plot_fcod(
            i_fcod,
            ax=ax,
            show_ci=True,
            xlabel=i_fcod["feature_name"],
            title=i_fcod["feature_name"],
        )

    fig.suptitle(f"Outcome FCODs by Feature (α={alpha})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig
    return


@app.cell
def _(N_COLS, alpha, fcod_results, math, plot_stacked_fcod, plt):
    def _():
        n_features = len(fcod_results)
        n_rows = math.ceil(n_features / N_COLS)
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
        for i in range(n_features, n_rows * N_COLS):
            axes.flatten()[i].set_visible(False)

        for idx, (feature_name, fcod) in enumerate(fcod_results.items()):
            ax = axes.flatten()[idx]
            plot_stacked_fcod(
                fcod,
                ax=ax,
                xlabel=fcod["feature_name"],
                title=fcod["feature_name"],
            )

        fig.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_results_ci, plot_multi_feature_fcod, plt):
    def _():
        fig = plot_multi_feature_fcod(
            fcod_results_ci,
            outcomes=["SC", "SI"],
            n_cols=4,
            show_ci=True,
            figsize_per_plot=(6, 4),
            share_y=False,
        )
        fig.suptitle(f"SC and SI Rates Across Features (α={alpha})", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_results_ci, np, plot_uncertainty_zones, plt):
    def _():
        fig, ax = plt.subplots(figsize=(12, 5))

        plot_uncertainty_zones(
            fcod_results_ci["age"],
            ax=ax,
            safe_threshold=0.7,
            uncertain_threshold=0.3,
            title=f"Decision Zones: Age (α={alpha})",
        )

        age_grid = fcod_results_ci["age"]["grid"]
        age_sc = fcod_results_ci["age"]["SC"]
        safe_idx = np.where(age_sc >= 0.7)[0]
        if len(safe_idx) > 0:
            safe_start = age_grid[safe_idx[0]]
            print(f"Safe zone (SC >= 70%) begins at Age ~= {safe_start:.0f}")
        else:
            print("No region achieves 70% SC threshold")

        return fig

    _()
    return


@app.cell
def _(X_test, alpha, math, np, plt, prediction_sets, y_test):
    from conformalpy.plots import plot_outcome_distribution_by_category

    cat_features = {
        "workclass":      {"label": "Workclass",      "top_n": None},
        "marital-status": {"label": "Marital Status", "top_n": None},
        "occupation":     {"label": "Occupation",     "top_n": 8},
        "relationship":   {"label": "Relationship",   "top_n": None},
        "race":           {"label": "Race",           "top_n": None},
        "sex":            {"label": "Sex",            "top_n": None},
        "native-country": {"label": "Native Country", "top_n": 10},
    }
    N_CAT_COLS = 3

    def _():
        n_cat = len(cat_features)
        n_cat_rows = math.ceil(n_cat / N_CAT_COLS)
        fig, axes = plt.subplots(n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, n_cat_rows * 5))
        for i in range(n_cat, n_cat_rows * N_CAT_COLS):
            axes.flatten()[i].set_visible(False)

        for idx, (feature_name, config) in enumerate(cat_features.items()):
            ax = axes.flatten()[idx]
            plot_outcome_distribution_by_category(
                X_test[feature_name].values,
                prediction_sets,
                np.asarray(y_test),
                ax=ax,
                category_name=config["label"],
                top_n=config["top_n"],
                sort_by="sc_rate",
                ascending=False,
            )

        fig.suptitle(
            f"Outcome Distribution by Categorical Features (α={alpha})",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        return fig

    _()
    return N_CAT_COLS, cat_features, plot_outcome_distribution_by_category


@app.cell
def _(
    N_CAT_COLS,
    N_COLS,
    OmegaConf,
    X_test,
    alpha,
    cat_features,
    cfg,
    data_path,
    fcod_results,
    fcod_results_ci,
    math,
    mlflow,
    mo,
    np,
    plot_fcod,
    plot_multi_feature_fcod,
    plot_outcome_distribution_by_category,
    plot_stacked_fcod,
    plot_uncertainty_zones,
    plt,
    prediction_sets,
    y_test,
):
    def _():
        from dslib.tracking import tracked_run

        config_dict = OmegaConf.to_container(cfg, resolve=True)

        with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="adult-fcod-analysis"):
            # FCOD line plots with CI
            _n_feat = len(fcod_results_ci)
            _n_rows = math.ceil(_n_feat / N_COLS)
            fig_ci, axes = plt.subplots(_n_rows, N_COLS, figsize=(N_COLS * 5, _n_rows * 4))
            for i in range(_n_feat, _n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, fcod) in enumerate(fcod_results_ci.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_ci.suptitle(f"Outcome FCODs by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_ci.tight_layout()
            mlflow.log_figure(fig_ci, "fcod_with_ci.png")
            plt.close(fig_ci)

            # Stacked area plots
            fig_stacked, axes = plt.subplots(_n_rows, N_COLS, figsize=(N_COLS * 5, _n_rows * 4))
            for i in range(_n_feat, _n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, fcod) in enumerate(fcod_results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_stacked.tight_layout()
            mlflow.log_figure(fig_stacked, "fcod_stacked.png")
            plt.close(fig_stacked)

            # SC + SI multi-feature grid
            fig_sc_si = plot_multi_feature_fcod(
                fcod_results_ci, outcomes=["SC", "SI"], n_cols=4,
                show_ci=True, figsize_per_plot=(6, 4),
            )
            fig_sc_si.suptitle(f"SC and SI Rates (α={alpha})", fontsize=14, y=1.02)
            fig_sc_si.tight_layout()
            mlflow.log_figure(fig_sc_si, "fcod_sc_si_grid.png")
            plt.close(fig_sc_si)

            # Uncertainty zones: age
            fig_zones, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_results_ci["age"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: Age (α={alpha})",
            )
            mlflow.log_figure(fig_zones, "uncertainty_zones_age.png")
            plt.close(fig_zones)

            # Categorical outcome distributions
            _n_cat = len(cat_features)
            _n_cat_rows = math.ceil(_n_cat / N_CAT_COLS)
            fig_cat, axes = plt.subplots(_n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, _n_cat_rows * 5))
            for i in range(_n_cat, _n_cat_rows * N_CAT_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, config) in enumerate(cat_features.items()):
                ax = axes.flatten()[idx]
                plot_outcome_distribution_by_category(
                    X_test[feature_name].values,
                    prediction_sets,
                    np.asarray(y_test),
                    ax=ax,
                    category_name=config["label"],
                    top_n=config["top_n"],
                    sort_by="sc_rate",
                    ascending=False,
                )
            fig_cat.suptitle(
                f"Outcome Distribution by Categorical Features (α={alpha})",
                fontsize=14, y=1.02,
            )
            fig_cat.tight_layout()
            mlflow.log_figure(fig_cat, "outcome_by_category.png")
            plt.close(fig_cat)

        return mo.md("FCOD plots logged to MLflow")

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
