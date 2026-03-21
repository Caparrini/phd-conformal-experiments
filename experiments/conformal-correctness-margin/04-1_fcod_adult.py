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
        "capital-gain":   {"label": "Capital Gain ($)",   "clip": (1, float(X_test["capital-gain"].quantile(0.99)))},
        "capital-loss":   {"label": "Capital Loss ($)",   "clip": (1, float(X_test["capital-loss"].quantile(0.99)))},
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


# === Per-class FCOD computation ===


@app.cell
def _(
    X_test,
    cfg,
    compute_fcod_smoothed,
    compute_fcod_with_ci,
    fcod_features,
    mo,
    np,
    prediction_sets,
    y_test,
):
    _y_arr = np.asarray(y_test)
    unique_classes = np.unique(_y_arr)

    fcod_by_class = {}
    fcod_ci_by_class = {}

    for _cls in unique_classes:
        _cls_mask = _y_arr == _cls
        _X_cls = X_test[_cls_mask]
        _ps_cls = [prediction_sets[i] for i, m in enumerate(_cls_mask) if m]
        _y_cls = _y_arr[_cls_mask]

        fcod_by_class[_cls] = {}
        fcod_ci_by_class[_cls] = {}

        for _feat, _config in fcod_features.items():
            _fv = _X_cls[_feat].values
            if _config["clip"] is not None:
                _lo, _hi = _config["clip"]
                _fmask = (_fv >= _lo) & (_fv <= _hi)
                _fv_viz = _fv[_fmask]
                _ps_viz = [_ps_cls[i] for i in range(len(_ps_cls)) if _fmask[i]]
                _y_viz = _y_cls[_fmask]
            else:
                _fv_viz, _ps_viz, _y_viz = _fv, _ps_cls, _y_cls

            _fcod = compute_fcod_smoothed(
                _fv_viz, _ps_viz, _y_viz, n_grid=50, percentile_range=(5, 95)
            )
            _fcod["feature_name"] = _config["label"]
            fcod_by_class[_cls][_feat] = _fcod

            _fcod_ci = compute_fcod_with_ci(
                _fv_viz, _ps_viz, _y_viz,
                n_bootstrap=100, n_grid=30, confidence_level=0.95,
                random_state=cfg.experiment.seed,
            )
            _fcod_ci["feature_name"] = _config["label"]
            fcod_ci_by_class[_cls][_feat] = _fcod_ci

    mo.md(f"Computed per-class FCODs for {len(unique_classes)} classes")
    return fcod_by_class, fcod_ci_by_class, unique_classes


@app.cell
def _(N_COLS, alpha, fcod_ci_by_class, math, plot_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_ci_by_class[cls]
            n_features = len(results)
            n_rows = math.ceil(n_features / N_COLS)
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
            for i in range(n_features, n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (fname, fcod) in enumerate(results.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome FCODs — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(N_COLS, alpha, fcod_by_class, math, plot_stacked_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_by_class[cls]
            n_features = len(results)
            n_rows = math.ceil(n_features / N_COLS)
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
            for i in range(n_features, n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (fname, fcod) in enumerate(results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome Distribution — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(alpha, fcod_ci_by_class, plot_multi_feature_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig = plot_multi_feature_fcod(
                fcod_ci_by_class[cls],
                outcomes=["SC", "SI"],
                n_cols=4,
                show_ci=True,
                figsize_per_plot=(6, 4),
                share_y=False,
            )
            fig.suptitle(f"SC and SI Rates — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(alpha, fcod_ci_by_class, plot_uncertainty_zones, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_ci_by_class[cls]["age"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: Age — Class {cls} (α={alpha})",
            )
            plt.tight_layout()

    _()
    return


@app.cell
def _(
    N_CAT_COLS,
    X_test,
    alpha,
    cat_features,
    math,
    np,
    plot_outcome_distribution_by_category,
    plt,
    prediction_sets,
    unique_classes,
    y_test,
):
    def _():
        for cls in unique_classes:
            _y_arr = np.asarray(y_test)
            _cls_mask = _y_arr == cls
            _X_cls = X_test[_cls_mask]
            _ps_cls = [prediction_sets[i] for i, m in enumerate(_cls_mask) if m]
            _y_cls = _y_arr[_cls_mask]

            n_cat = len(cat_features)
            n_cat_rows = math.ceil(n_cat / N_CAT_COLS)
            fig, axes = plt.subplots(n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, n_cat_rows * 5))
            for i in range(n_cat, n_cat_rows * N_CAT_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, config) in enumerate(cat_features.items()):
                ax = axes.flatten()[idx]
                plot_outcome_distribution_by_category(
                    _X_cls[feature_name].values,
                    _ps_cls,
                    _y_cls,
                    ax=ax,
                    category_name=config["label"],
                    top_n=config["top_n"],
                    sort_by="sc_rate",
                    ascending=False,
                )
            fig.suptitle(
                f"Outcome Distribution by Category — Class {cls} (α={alpha})",
                fontsize=14, y=1.02,
            )
            plt.tight_layout()

    _()
    return


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
    fcod_by_class,
    fcod_ci_by_class,
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
    unique_classes,
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
            mlflow.log_figure(fig_ci, "fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
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
                mlflow.log_figure(_main_ax.figure, f"fcod_histogram/{_feature_name}.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_main_ax.figure)

            # Stacked area plots
            fig_stacked, axes = plt.subplots(_n_rows, N_COLS, figsize=(N_COLS * 5, _n_rows * 4))
            for i in range(_n_feat, _n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, fcod) in enumerate(fcod_results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_stacked.tight_layout()
            mlflow.log_figure(fig_stacked, "fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_stacked)

            # SC + SI multi-feature grid
            fig_sc_si = plot_multi_feature_fcod(
                fcod_results_ci, outcomes=["SC", "SI"], n_cols=4,
                show_ci=True, figsize_per_plot=(6, 4),
            )
            fig_sc_si.suptitle(f"SC and SI Rates (α={alpha})", fontsize=14, y=1.02)
            fig_sc_si.tight_layout()
            mlflow.log_figure(fig_sc_si, "fcod_sc_si_grid.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_sc_si)

            # Uncertainty zones: age
            fig_zones, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_results_ci["age"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: Age (α={alpha})",
            )
            mlflow.log_figure(fig_zones, "uncertainty_zones_age.png", save_kwargs={"bbox_inches": "tight"})
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
            mlflow.log_figure(fig_cat, "outcome_by_category.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_cat)

            # === Per-class plots ===
            for _cls in unique_classes:
                _prefix = f"by_class/class_{_cls}"
                _cls_fcod_ci = fcod_ci_by_class[_cls]
                _cls_fcod_smooth = fcod_by_class[_cls]

                _y_arr = np.asarray(y_test)
                _cls_mask = _y_arr == _cls
                _X_cls = X_test[_cls_mask]
                _ps_cls = [prediction_sets[i] for i, m in enumerate(_cls_mask) if m]
                _y_cls = _y_arr[_cls_mask]

                # FCOD with CI
                _n_feat_cls = len(_cls_fcod_ci)
                _n_rows_cls = math.ceil(_n_feat_cls / N_COLS)
                _fig_ci, _axes = plt.subplots(_n_rows_cls, N_COLS, figsize=(N_COLS * 5, _n_rows_cls * 4))
                for i in range(_n_feat_cls, _n_rows_cls * N_COLS):
                    _axes.flatten()[i].set_visible(False)
                for idx, (feat, fcod) in enumerate(_cls_fcod_ci.items()):
                    plot_fcod(fcod, ax=_axes.flatten()[idx], show_ci=True,
                              xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_ci.suptitle(f"Outcome FCODs — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_ci.tight_layout()
                mlflow.log_figure(_fig_ci, f"{_prefix}/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_ci)

                # Per-feature histogram
                for _feat, _fcod in _cls_fcod_ci.items():
                    _main_ax, _density_ax = plot_fcod(
                        _fcod,
                        show_ci=True,
                        show_density=True,
                        density_type="histogram",
                        feature_values=_X_cls[_feat].values,
                        xlabel=_fcod["feature_name"],
                        title=_fcod["feature_name"],
                    )
                    mlflow.log_figure(_main_ax.figure, f"{_prefix}/fcod_histogram/{_feat}.png", save_kwargs={"bbox_inches": "tight"})
                    plt.close(_main_ax.figure)

                # Stacked
                _fig_stacked, _axes = plt.subplots(_n_rows_cls, N_COLS, figsize=(N_COLS * 5, _n_rows_cls * 4))
                for i in range(_n_feat_cls, _n_rows_cls * N_COLS):
                    _axes.flatten()[i].set_visible(False)
                for idx, (feat, fcod) in enumerate(_cls_fcod_smooth.items()):
                    plot_stacked_fcod(fcod, ax=_axes.flatten()[idx],
                                      xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_stacked.suptitle(f"Outcome Distribution — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_stacked.tight_layout()
                mlflow.log_figure(_fig_stacked, f"{_prefix}/fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_stacked)

                # SC/SI grid
                _fig_sc_si = plot_multi_feature_fcod(
                    _cls_fcod_ci, outcomes=["SC", "SI"], n_cols=4,
                    show_ci=True, figsize_per_plot=(6, 4),
                )
                _fig_sc_si.suptitle(f"SC and SI Rates — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_sc_si.tight_layout()
                mlflow.log_figure(_fig_sc_si, f"{_prefix}/fcod_sc_si_grid.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_sc_si)

                # Uncertainty zones: age
                _fig_zones, _ax = plt.subplots(figsize=(12, 5))
                plot_uncertainty_zones(
                    _cls_fcod_ci["age"], ax=_ax,
                    safe_threshold=0.7, uncertain_threshold=0.3,
                    title=f"Decision Zones: Age — Class {_cls} (α={alpha})",
                )
                mlflow.log_figure(_fig_zones, f"{_prefix}/uncertainty_zones_age.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_zones)

                # Categorical
                _n_cat = len(cat_features)
                _n_cat_rows = math.ceil(_n_cat / N_CAT_COLS)
                _fig_cat, _axes_cat = plt.subplots(_n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, _n_cat_rows * 5))
                for i in range(_n_cat, _n_cat_rows * N_CAT_COLS):
                    _axes_cat.flatten()[i].set_visible(False)
                for idx, (feature_name, config) in enumerate(cat_features.items()):
                    _ax_cat = _axes_cat.flatten()[idx]
                    plot_outcome_distribution_by_category(
                        _X_cls[feature_name].values,
                        _ps_cls,
                        _y_cls,
                        ax=_ax_cat,
                        category_name=config["label"],
                        top_n=config["top_n"],
                        sort_by="sc_rate",
                        ascending=False,
                    )
                _fig_cat.suptitle(
                    f"Outcome Distribution by Category — Class {_cls} (α={alpha})",
                    fontsize=14, y=1.02,
                )
                _fig_cat.tight_layout()
                mlflow.log_figure(_fig_cat, f"{_prefix}/outcome_by_category.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_cat)

        return mo.md("FCOD plots logged to MLflow")

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
