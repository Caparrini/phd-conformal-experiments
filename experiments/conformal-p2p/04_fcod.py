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
        cfg = compose(config_name="config")

    alpha = cfg.conformal.alpha
    return (
        EXPERIMENT_DIR,
        OmegaConf,
        alpha,
        cfg,
        compute_fcod_smoothed,
        compute_fcod_with_ci,
        json,
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

    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    mo.md(f"Test set: {len(X_test):,} samples")
    return X_test, data_path, y_test


@app.cell
def _(cfg, json, mlflow, mo):
    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'conformal-evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0].run_id

    artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/outputs/prediction_sets.json")
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
    prediction_sets,
    y_test,
):
    fcod_features = {
        "fico_n": {"label": "FICO Score", "clip": None},
        "dti_n": {"label": "DTI Ratio", "clip": (0, 100)},
        "loan_amnt": {"label": "Loan Amount ($)", "clip": None},
        "revenue": {"label": "Annual Revenue ($)", "clip": (0, float(X_test["revenue"].quantile(0.99)))},
    }

    fcod_results = {}
    fcod_results_ci = {}

    for feature_name, config in fcod_features.items():
        feature_values = X_test[feature_name].values

        if config["clip"] is not None:
            lo, hi = config["clip"]
            mask = (feature_values >= lo) & (feature_values <= hi)
            feature_values_viz = feature_values[mask]
            pred_sets_viz = [prediction_sets[i] for i in range(len(prediction_sets)) if mask[i]]
            y_test_viz = y_test.values[mask]
        else:
            feature_values_viz = feature_values
            pred_sets_viz = prediction_sets
            y_test_viz = y_test.values

        # Smoothed FCOD
        fcod = compute_fcod_smoothed(
            feature_values_viz, pred_sets_viz, y_test_viz,
            n_grid=50, percentile_range=(5, 95),
        )
        fcod["feature_name"] = config["label"]
        fcod_results[feature_name] = fcod

        # FCOD with CI
        fcod_ci = compute_fcod_with_ci(
            feature_values_viz, pred_sets_viz, y_test_viz,
            n_bootstrap=100, n_grid=30, confidence_level=0.95,
            random_state=cfg.experiment.seed,
        )
        fcod_ci["feature_name"] = config["label"]
        fcod_results_ci[feature_name] = fcod_ci

    mo.md(f"Computed FCODs for {len(fcod_features)} features")
    return fcod_results, fcod_results_ci


@app.cell
def _(alpha, fcod_results_ci, plot_fcod, plt):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
def _(alpha, fcod_results, plot_stacked_fcod, plt):
    def _():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
            n_cols=2,
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
def _(alpha, fcod_results_ci, mo, np, plot_uncertainty_zones, plt):
    def _():
        fig, ax = plt.subplots(figsize=(12, 5))

        plot_uncertainty_zones(
            fcod_results_ci["fico_n"],
            ax=ax,
            safe_threshold=0.7,
            uncertain_threshold=0.3,
            title=f"Decision Zones: FICO Score (α={alpha})",
        )

        # Find where safe zone begins
        fico_grid = fcod_results_ci["fico_n"]["grid"]
        fico_sc = fcod_results_ci["fico_n"]["SC"]
        safe_idx = np.where(fico_sc >= 0.7)[0]
        if len(safe_idx) > 0:
            safe_start = fico_grid[safe_idx[0]]
            mo.md(f"Safe zone (SC ≥ 70%) begins at FICO ≈ **{safe_start:.0f}**")
        else:
            mo.md("No region achieves 70% SC threshold")
        return fig


    _()
    return


@app.cell
def _(X_test, alpha, plt, prediction_sets, y_test):
    from conformalpy.plots import plot_outcome_distribution_by_category

    cat_features = {
            "emp_length": {"label": "Employment Length", "top_n": None},
            "purpose": {"label": "Loan Purpose", "top_n": 10},
            "home_ownership_n": {"label": "Home Ownership", "top_n": None},
            "addr_state": {"label": "State", "top_n": 15},
    }

    def _():


        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (feature_name, config) in enumerate(cat_features.items()):
            ax = axes.flatten()[idx]
            plot_outcome_distribution_by_category(
                X_test[feature_name].values,
                prediction_sets,
                y_test.values,
                ax=ax,
                category_name=config["label"],
                top_n=config["top_n"],
                sort_by="sc_rate",
                ascending=False,
            )

        fig.suptitle(f"Outcome Distribution by Categorical Features (α={alpha})", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig


    _()
    return cat_features, plot_outcome_distribution_by_category


@app.cell
def _(
    OmegaConf,
    X_test,
    alpha,
    cat_features,
    cfg,
    data_path,
    fcod_results,
    fcod_results_ci,
    mlflow,
    mo,
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

        with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="fcod-analysis"):
            # FCOD line plots with CI
            fig_ci, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feature_name, fcod) in enumerate(fcod_results_ci.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_ci.suptitle(f"Outcome FCODs by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_ci.tight_layout()
            mlflow.log_figure(fig_ci, "fcod_with_ci.png")
            plt.close(fig_ci)

            # Per-feature FCOD with histogram density panel
            for _feature_name, _fcod in fcod_results_ci.items():
                _main_ax, _density_ax = plot_fcod(
                    _fcod,
                    show_ci=True,
                    show_density=True,
                    density_type="histogram",
                    feature_values=X_test[_feature_name].values,
                    xlabel=_fcod["feature_name"],
                    title=_fcod["feature_name"],
                )
                mlflow.log_figure(_main_ax.figure, f"fcod_histogram/{_feature_name}.png")
                plt.close(_main_ax.figure)

            # Stacked area plots
            fig_stacked, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feature_name, fcod) in enumerate(fcod_results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_stacked.tight_layout()
            mlflow.log_figure(fig_stacked, "fcod_stacked.png")
            plt.close(fig_stacked)

            # SC + SI multi-feature grid
            fig_sc_si = plot_multi_feature_fcod(
                fcod_results_ci, outcomes=["SC", "SI"], n_cols=2,
                show_ci=True, figsize_per_plot=(6, 4),
            )
            fig_sc_si.suptitle(f"SC and SI Rates (α={alpha})", fontsize=14, y=1.02)
            fig_sc_si.tight_layout()
            mlflow.log_figure(fig_sc_si, "fcod_sc_si_grid.png")
            plt.close(fig_sc_si)

            # Uncertainty zones FICO
            fig_zones, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(fcod_results_ci["fico_n"], ax=ax,
                                   safe_threshold=0.7, uncertain_threshold=0.3,
                                   title=f"Decision Zones: FICO Score (α={alpha})")
            mlflow.log_figure(fig_zones, "uncertainty_zones_fico.png")
            plt.close(fig_zones)

            # Categorical outcome distributions
            fig_cat, axes = plt.subplots(2, 2, figsize=(16, 12))
            for idx, (feature_name, config) in enumerate(cat_features.items()):
                ax = axes.flatten()[idx]
                plot_outcome_distribution_by_category(
                    X_test[feature_name].values,
                    prediction_sets,
                    y_test.values,
                    ax=ax,
                    category_name=config["label"],
                    top_n=config["top_n"],
                    sort_by="sc_rate",
                    ascending=False,
                )
            fig_cat.suptitle(f"Outcome Distribution by Categorical Features (α={alpha})", fontsize=14, y=1.02)
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
