import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import mlflow
    import json

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    from conformalpy.fcod import (
        compute_fcod_smoothed,
        compute_fcod_with_ci,
        merge_uncertain_outcomes,
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
        merge_uncertain_outcomes,
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

    fcod_results_merged = {k: merge_uncertain_outcomes(v) for k, v in fcod_results.items()}
    fcod_results_ci_merged = {k: merge_uncertain_outcomes(v) for k, v in fcod_results_ci.items()}

    mo.md(f"Computed FCODs for {len(fcod_features)} features")
    return fcod_features, fcod_results, fcod_results_ci, fcod_results_merged, fcod_results_ci_merged


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
def _(X_test, cfg, compute_fcod_smoothed, compute_fcod_with_ci, fcod_features, mo, np, prediction_sets, y_test):
    _y_arr = y_test.values
    unique_classes = np.unique(_y_arr)

    fcod_by_class: dict = {}
    fcod_ci_by_class: dict = {}

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
                _clip_mask = (_fv >= _lo) & (_fv <= _hi)
                _fv_viz = _fv[_clip_mask]
                _ps_viz = [_ps_cls[i] for i, m in enumerate(_clip_mask) if m]
                _y_viz = _y_cls[_clip_mask]
            else:
                _fv_viz = _fv
                _ps_viz = _ps_cls
                _y_viz = _y_cls

            _fcod = compute_fcod_smoothed(
                _fv_viz, _ps_viz, _y_viz,
                n_grid=50, percentile_range=(5, 95),
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

    fcod_by_class_merged = {
        cls: {k: merge_uncertain_outcomes(v) for k, v in d.items()}
        for cls, d in fcod_by_class.items()
    }
    fcod_ci_by_class_merged = {
        cls: {k: merge_uncertain_outcomes(v) for k, v in d.items()}
        for cls, d in fcod_ci_by_class.items()
    }

    mo.md(f"Computed per-class FCODs for {len(unique_classes)} classes")
    return fcod_by_class, fcod_ci_by_class, fcod_by_class_merged, fcod_ci_by_class_merged, unique_classes


@app.cell
def _(alpha, fcod_ci_by_class, plot_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_ci_by_class[cls]
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feat, fcod) in enumerate(results.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome FCODs — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_by_class, plot_stacked_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_by_class[cls]
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feat, fcod) in enumerate(results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome Distribution — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_ci_by_class, plot_multi_feature_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig = plot_multi_feature_fcod(
                fcod_ci_by_class[cls],
                outcomes=["SC", "SI"],
                n_cols=2,
                show_ci=True,
                figsize_per_plot=(6, 4),
            )
            fig.suptitle(f"SC and SI Rates — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_ci_by_class, plot_uncertainty_zones, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_ci_by_class[cls]["fico_n"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: FICO Score — Class {cls} (α={alpha})",
            )
        return fig

    _()
    return


@app.cell
def _(X_test, alpha, cat_features, plot_outcome_distribution_by_category, plt, prediction_sets, unique_classes, y_test):
    def _():
        _y_arr_cls = y_test.values
        for cls in unique_classes:
            _cls_mask = _y_arr_cls == cls
            _X_cls = X_test[_cls_mask]
            _ps_cls = [prediction_sets[i] for i, m in enumerate(_cls_mask) if m]
            _y_cls = _y_arr_cls[_cls_mask]

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
            fig.suptitle(f"Outcome Distribution by Categorical Features — Class {cls} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(
    OmegaConf,
    X_test,
    alpha,
    cat_features,
    cfg,
    data_path,
    fcod_by_class,
    fcod_by_class_merged,
    fcod_ci_by_class,
    fcod_ci_by_class_merged,
    fcod_results,
    fcod_results_ci,
    fcod_results_ci_merged,
    fcod_results_merged,
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

        with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="fcod-analysis"):
            # FCOD line plots with CI
            fig_ci, axes = plt.subplots(2, 2, figsize=(14, 10))
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
            fig_stacked, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feature_name, fcod) in enumerate(fcod_results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_stacked.tight_layout()
            mlflow.log_figure(fig_stacked, "fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_stacked)

            # SC + SI multi-feature grid
            fig_sc_si = plot_multi_feature_fcod(
                fcod_results_ci, outcomes=["SC", "SI"], n_cols=2,
                show_ci=True, figsize_per_plot=(6, 4),
            )
            fig_sc_si.suptitle(f"SC and SI Rates (α={alpha})", fontsize=14, y=1.02)
            fig_sc_si.tight_layout()
            mlflow.log_figure(fig_sc_si, "fcod_sc_si_grid.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_sc_si)

            # Uncertainty zones FICO
            fig_zones, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(fcod_results_ci["fico_n"], ax=ax,
                                   safe_threshold=0.7, uncertain_threshold=0.3,
                                   title=f"Decision Zones: FICO Score (α={alpha})")
            mlflow.log_figure(fig_zones, "uncertainty_zones_fico.png", save_kwargs={"bbox_inches": "tight"})
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
            mlflow.log_figure(fig_cat, "outcome_by_category.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_cat)

            # === Per-class plots ===
            _y_arr = y_test.values
            for _cls in unique_classes:
                _prefix = f"by_class/class_{_cls}"
                _cls_fcod_ci = fcod_ci_by_class[_cls]
                _cls_fcod_smooth = fcod_by_class[_cls]

                _cls_mask = _y_arr == _cls
                _X_cls = X_test[_cls_mask]
                _ps_cls = [prediction_sets[i] for i, m in enumerate(_cls_mask) if m]
                _y_cls = _y_arr[_cls_mask]

                # FCOD with CI
                _fig_ci, _axes = plt.subplots(2, 2, figsize=(14, 10))
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
                _fig_stacked, _axes = plt.subplots(2, 2, figsize=(14, 10))
                for idx, (feat, fcod) in enumerate(_cls_fcod_smooth.items()):
                    plot_stacked_fcod(fcod, ax=_axes.flatten()[idx],
                                      xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_stacked.suptitle(f"Outcome Distribution — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_stacked.tight_layout()
                mlflow.log_figure(_fig_stacked, f"{_prefix}/fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_stacked)

                # SC/SI grid
                _fig_sc_si = plot_multi_feature_fcod(
                    _cls_fcod_ci, outcomes=["SC", "SI"], n_cols=2,
                    show_ci=True, figsize_per_plot=(6, 4),
                )
                _fig_sc_si.suptitle(f"SC and SI Rates — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_sc_si.tight_layout()
                mlflow.log_figure(_fig_sc_si, f"{_prefix}/fcod_sc_si_grid.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_sc_si)

                # Uncertainty zones: FICO score
                _fig_zones, _ax = plt.subplots(figsize=(12, 5))
                plot_uncertainty_zones(
                    _cls_fcod_ci["fico_n"], ax=_ax,
                    safe_threshold=0.7, uncertain_threshold=0.3,
                    title=f"Decision Zones: FICO Score — Class {_cls} (α={alpha})",
                )
                mlflow.log_figure(_fig_zones, f"{_prefix}/uncertainty_zones_fico.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_zones)

                # Categorical outcome distributions
                _fig_cat, _axes = plt.subplots(2, 2, figsize=(16, 12))
                for idx, (feature_name, config) in enumerate(cat_features.items()):
                    _ax = _axes.flatten()[idx]
                    plot_outcome_distribution_by_category(
                        _X_cls[feature_name].values,
                        _ps_cls,
                        _y_cls,
                        ax=_ax,
                        category_name=config["label"],
                        top_n=config["top_n"],
                        sort_by="sc_rate",
                        ascending=False,
                    )
                _fig_cat.suptitle(f"Outcome Distribution by Categorical Features — Class {_cls} (α={alpha})", fontsize=14, y=1.02)
                _fig_cat.tight_layout()
                mlflow.log_figure(_fig_cat, f"{_prefix}/outcome_by_category.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_cat)

            # === Stacked FCOD by class (per feature) ===
            _fv_by_class = {}
            for _cls in unique_classes:
                _cls_mask = _y_arr == _cls
                _X_cls = X_test[_cls_mask]
                _fv_by_class[_cls] = {
                    feat: _X_cls[feat].values for feat in fcod_by_class[_cls]
                }

            _features = list(fcod_by_class[unique_classes[0]].keys())
            for _feat in _features:
                _fcod_feat = {c: fcod_by_class[c][_feat] for c in unique_classes}
                _fv_feat = {c: _fv_by_class[c][_feat] for c in unique_classes}
                _fig_stacked_cls = plot_stacked_fcod(
                    {},
                    fcod_by_class=_fcod_feat,
                    feature_values_by_class=_fv_feat,
                    class_names={c: f"Class {c}" for c in unique_classes},
                    figsize=(6, 4),
                    show_density=True,
                    show_std=False,
                    title=_feat,
                )
                _fig_stacked_cls.suptitle(
                    f"Stacked FCOD by Class — {_feat} (α={alpha})",
                    fontsize=14, y=1.01,
                )
                mlflow.log_figure(
                    _fig_stacked_cls,
                    f"by_class_stacked/{_feat}.png",
                    save_kwargs={"bbox_inches": "tight"},
                )
                plt.close(_fig_stacked_cls)

            # === Merged uncertain outcomes (TS0+TS1 → TS) ===

            # Merged FCOD line plots with CI
            fig_ci_m, axes_m = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feature_name, fcod) in enumerate(fcod_results_ci_merged.items()):
                plot_fcod(fcod, ax=axes_m.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_ci_m.suptitle(f"Outcome FCODs by Feature — Merged TS (α={alpha})", fontsize=14, y=1.02)
            fig_ci_m.tight_layout()
            mlflow.log_figure(fig_ci_m, "merged/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_ci_m)

            # Merged stacked area plots (grid)
            fig_stacked_m, axes_m = plt.subplots(2, 2, figsize=(14, 10))
            for idx, (feature_name, fcod) in enumerate(fcod_results_merged.items()):
                plot_stacked_fcod(fcod, ax=axes_m.flatten()[idx],
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked_m.suptitle(f"Outcome Distribution — Merged TS (α={alpha})", fontsize=14, y=1.02)
            fig_stacked_m.tight_layout()
            mlflow.log_figure(fig_stacked_m, "merged/fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_stacked_m)

            # Merged per-feature stacked + density
            for _feature_name, _fcod in fcod_results_merged.items():
                _result = plot_stacked_fcod(
                    _fcod,
                    show_density=True,
                    density_type="histogram",
                    feature_values=X_test[_feature_name].values,
                    xlabel=_fcod["feature_name"],
                    title=_fcod["feature_name"],
                )
                _fig_sd = _result[0].figure if isinstance(_result, tuple) else _result.figure
                mlflow.log_figure(_fig_sd, f"merged/fcod_stacked_density/{_feature_name}.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_sd)

            # Merged per-feature FCOD + density
            for _feature_name, _fcod in fcod_results_ci_merged.items():
                _main_ax_m, _density_ax_m = plot_fcod(
                    _fcod,
                    show_ci=True,
                    show_density=True,
                    density_type="histogram",
                    feature_values=X_test[_feature_name].values,
                    xlabel=_fcod["feature_name"],
                    title=_fcod["feature_name"],
                )
                mlflow.log_figure(_main_ax_m.figure, f"merged/fcod_histogram/{_feature_name}.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_main_ax_m.figure)

            # Non-merged per-feature stacked + density
            for _feature_name, _fcod in fcod_results.items():
                _result = plot_stacked_fcod(
                    _fcod,
                    show_density=True,
                    density_type="histogram",
                    feature_values=X_test[_feature_name].values,
                    xlabel=_fcod["feature_name"],
                    title=_fcod["feature_name"],
                )
                _fig_sd = _result[0].figure if isinstance(_result, tuple) else _result.figure
                mlflow.log_figure(_fig_sd, f"fcod_stacked_density/{_feature_name}.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_sd)

            # Merged per-class artifacts
            _y_arr = y_test.values
            for _cls in unique_classes:
                _prefix_m = f"merged/by_class/class_{_cls}"
                _cls_fcod_ci_m = fcod_ci_by_class_merged[_cls]
                _cls_fcod_m = fcod_by_class_merged[_cls]
                _cls_mask = _y_arr == _cls
                _X_cls = X_test[_cls_mask]

                # Merged FCOD with CI
                _fig_ci_m2, _axes_m2 = plt.subplots(2, 2, figsize=(14, 10))
                for idx, (feat, fcod) in enumerate(_cls_fcod_ci_m.items()):
                    plot_fcod(fcod, ax=_axes_m2.flatten()[idx], show_ci=True,
                              xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_ci_m2.suptitle(f"Outcome FCODs — Class {_cls}, Merged TS (α={alpha})", fontsize=14, y=1.02)
                _fig_ci_m2.tight_layout()
                mlflow.log_figure(_fig_ci_m2, f"{_prefix_m}/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_ci_m2)

                # Merged stacked grid per class
                _fig_stacked_m2, _axes_m2 = plt.subplots(2, 2, figsize=(14, 10))
                for idx, (feat, fcod) in enumerate(_cls_fcod_m.items()):
                    plot_stacked_fcod(fcod, ax=_axes_m2.flatten()[idx],
                                      xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_stacked_m2.suptitle(f"Outcome Distribution — Class {_cls}, Merged TS (α={alpha})", fontsize=14, y=1.02)
                _fig_stacked_m2.tight_layout()
                mlflow.log_figure(_fig_stacked_m2, f"{_prefix_m}/fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_stacked_m2)

                # Merged per-feature stacked + density per class
                for _feat, _fcod in _cls_fcod_m.items():
                    _result = plot_stacked_fcod(
                        _fcod,
                        show_density=True,
                        density_type="histogram",
                        feature_values=_X_cls[_feat].values,
                        xlabel=_fcod["feature_name"],
                        title=_fcod["feature_name"],
                    )
                    _fig_sd = _result[0].figure if isinstance(_result, tuple) else _result.figure
                    mlflow.log_figure(_fig_sd, f"{_prefix_m}/fcod_stacked_density/{_feat}.png", save_kwargs={"bbox_inches": "tight"})
                    plt.close(_fig_sd)

        return mo.md("FCOD plots logged to MLflow")


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
