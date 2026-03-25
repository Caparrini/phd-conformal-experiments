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
        merge_uncertain_outcomes,
        plot_fcod,
        plot_stacked_fcod,
        plot_stacked_bars_by_class,
        plot_multi_feature_fcod,
        plot_uncertainty_zones,
    )

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="wine_quality")

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
        plot_stacked_bars_by_class,
        plot_stacked_fcod,
        plot_uncertainty_zones,
        plt,
    )


@app.cell
def _(EXPERIMENT_DIR, cfg, mo):
    from dslib.data_utils import stratified_split
    from sklearn.preprocessing import LabelEncoder

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

    X_train_raw, y_train_raw = splits["train"]
    X_test, y_test_raw = splits["test"]

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_test = le.transform(y_test_raw)

    class_names = {i: name for i, name in enumerate(le.classes_)}

    mo.md(f"Test set: {len(X_test):,} samples | Classes: {list(le.classes_)}")
    return X_test, class_names, data_path, le, y_test


@app.cell
def _(cfg, json, mlflow, mo):
    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'wine-conformal-evaluation'",
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
    # Roughly normal, bounded → no clip
    _no_clip = {"density", "ph", "alcohol"}

    # Right-skewed → (q01, q99)
    _both_clip = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                  "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "sulphates"}

    fcod_features = {}
    for col in cfg.data.features.numerical:
        label = col.replace("_", " ").title()
        if col in _no_clip:
            clip = None
        else:  # right-skewed features
            clip = (float(X_test[col].quantile(0.01)), float(X_test[col].quantile(0.99)))
        fcod_features[col] = {"label": label, "clip": clip}
    N_COLS = 4
    OUTCOMES = ["SC", "SI", "MC", "MU"]

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

    fcod_results_merged = {k: merge_uncertain_outcomes(v) for k, v in fcod_results.items()}
    fcod_results_ci_merged = {k: merge_uncertain_outcomes(v) for k, v in fcod_results_ci.items()}

    mo.md(f"Computed FCODs for {len(fcod_features)} features")
    return N_COLS, OUTCOMES, fcod_features, fcod_results, fcod_results_ci, fcod_results_merged, fcod_results_ci_merged


@app.cell
def _(N_COLS, OUTCOMES, alpha, fcod_results_ci, math, plot_fcod, plt):
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
            outcomes=OUTCOMES,
            show_ci=True,
            xlabel=i_fcod["feature_name"],
            title=i_fcod["feature_name"],
        )

    fig.suptitle(f"Outcome FCODs by Feature (α={alpha})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig
    return


@app.cell
def _(N_COLS, OUTCOMES, alpha, fcod_results, math, plot_stacked_fcod, plt):
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
                outcomes=OUTCOMES,
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
            outcomes=["SC", "MC"],
            n_cols=4,
            show_ci=True,
            figsize_per_plot=(6, 4),
            share_y=False,
        )
        fig.suptitle(f"SC and MC Rates Across Features (α={alpha})", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(alpha, fcod_results_ci, np, plot_uncertainty_zones, plt):
    def _():
        fig, ax = plt.subplots(figsize=(12, 5))

        plot_uncertainty_zones(
            fcod_results_ci["alcohol"],
            ax=ax,
            safe_threshold=0.7,
            uncertain_threshold=0.3,
            title=f"Decision Zones: Alcohol (α={alpha})",
        )

        alcohol_grid = fcod_results_ci["alcohol"]["grid"]
        alcohol_sc = fcod_results_ci["alcohol"]["SC"]
        safe_idx = np.where(alcohol_sc >= 0.7)[0]
        if len(safe_idx) > 0:
            safe_start = alcohol_grid[safe_idx[0]]
            print(f"Safe zone (SC >= 70%) begins at Alcohol ~= {safe_start:.2f}%")
        else:
            print("No region achieves 70% SC threshold")

        return fig

    _()
    return


@app.cell
def _(X_test, alpha, math, np, plt, prediction_sets, y_test):
    from conformalpy.plots import plot_outcome_distribution_by_category

    cat_features = {
        "wine_type": {"label": "Wine Type", "top_n": None},
    }
    N_CAT_COLS = 1

    def _():
        n_cat = len(cat_features)
        n_cat_rows = math.ceil(n_cat / N_CAT_COLS)
        fig, axes = plt.subplots(n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, n_cat_rows * 5))
        # axes is a single Axes when there's only one subplot — wrap for uniform iteration
        axes_flat = [axes] if n_cat == 1 else axes.flatten()
        for i in range(n_cat, n_cat_rows * N_CAT_COLS):
            axes_flat[i].set_visible(False)

        for idx, (feature_name, config) in enumerate(cat_features.items()):
            ax = axes_flat[idx]
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
    class_names,
    compute_fcod_smoothed,
    compute_fcod_with_ci,
    fcod_features,
    mo,
    np,
    prediction_sets,
    y_test,
    OUTCOMES,
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

    fcod_by_class_merged = {
        cls: {k: merge_uncertain_outcomes(v) for k, v in d.items()}
        for cls, d in fcod_by_class.items()
    }
    fcod_ci_by_class_merged = {
        cls: {k: merge_uncertain_outcomes(v) for k, v in d.items()}
        for cls, d in fcod_ci_by_class.items()
    }

    mo.md(f"Computed per-class FCODs for {len(unique_classes)} classes: {[class_names[c] for c in unique_classes]}")
    return fcod_by_class, fcod_ci_by_class, fcod_by_class_merged, fcod_ci_by_class_merged, unique_classes


@app.cell
def _(OUTCOMES, N_COLS, alpha, class_names, fcod_ci_by_class, math, plot_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_ci_by_class[cls]
            n_features = len(results)
            n_rows = math.ceil(n_features / N_COLS)
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
            for i in range(n_features, n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (fname, fcod) in enumerate(results.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], outcomes=OUTCOMES, show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome FCODs — Class {class_names[cls]} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(OUTCOMES, N_COLS, alpha, class_names, fcod_by_class, math, plot_stacked_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            results = fcod_by_class[cls]
            n_features = len(results)
            n_rows = math.ceil(n_features / N_COLS)
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
            for i in range(n_features, n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (fname, fcod) in enumerate(results.items()):
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx], outcomes=OUTCOMES,
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig.suptitle(f"Outcome Distribution — Class {class_names[cls]} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(alpha, class_names, fcod_ci_by_class, plot_multi_feature_fcod, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig = plot_multi_feature_fcod(
                fcod_ci_by_class[cls],
                outcomes=["SC", "MC"],
                n_cols=4,
                show_ci=True,
                figsize_per_plot=(6, 4),
                share_y=False,
            )
            fig.suptitle(f"SC and MC Rates — Class {class_names[cls]} (α={alpha})", fontsize=14, y=1.02)
            plt.tight_layout()

    _()
    return


@app.cell
def _(alpha, class_names, fcod_ci_by_class, plot_uncertainty_zones, plt, unique_classes):
    def _():
        for cls in unique_classes:
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_ci_by_class[cls]["alcohol"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: Alcohol — Class {class_names[cls]} (α={alpha})",
            )
            plt.tight_layout()

    _()
    return


@app.cell
def _(
    X_test,
    alpha,
    cat_features,
    class_names,
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
            # N_CAT_COLS = 1 for wine — single axes, wrap for uniform handling
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes_flat = [axes]
            for idx, (feature_name, config) in enumerate(cat_features.items()):
                ax = axes_flat[idx]
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
                f"Outcome Distribution by Category — Class {class_names[cls]} (α={alpha})",
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
    OUTCOMES,
    X_test,
    alpha,
    cat_features,
    cfg,
    class_names,
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
    plot_stacked_bars_by_class,
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

        with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="wine-fcod-analysis"):
            # FCOD line plots with CI
            _n_feat = len(fcod_results_ci)
            _n_rows = math.ceil(_n_feat / N_COLS)
            fig_ci, axes = plt.subplots(_n_rows, N_COLS, figsize=(N_COLS * 5, _n_rows * 4))
            for i in range(_n_feat, _n_rows * N_COLS):
                axes.flatten()[i].set_visible(False)
            for idx, (feature_name, fcod) in enumerate(fcod_results_ci.items()):
                plot_fcod(fcod, ax=axes.flatten()[idx], outcomes=OUTCOMES, show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_ci.suptitle(f"Outcome FCODs by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_ci.tight_layout()
            mlflow.log_figure(fig_ci, "fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_ci)

            # Per-feature FCOD with histogram density panel
            for _feature_name, _fcod in fcod_results_ci.items():
                _main_ax, _density_ax = plot_fcod(
                    _fcod,
                    outcomes=OUTCOMES,
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
                plot_stacked_fcod(fcod, ax=axes.flatten()[idx], outcomes=OUTCOMES,
                                  xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_stacked.suptitle(f"Outcome Distribution by Feature (α={alpha})", fontsize=14, y=1.02)
            fig_stacked.tight_layout()
            mlflow.log_figure(fig_stacked, "fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_stacked)

            # SC + MC multi-feature grid
            fig_sc_mc = plot_multi_feature_fcod(
                fcod_results_ci, outcomes=["SC", "MC"], n_cols=4,
                show_ci=True, figsize_per_plot=(6, 4),
            )
            fig_sc_mc.suptitle(f"SC and MC Rates (α={alpha})", fontsize=14, y=1.02)
            fig_sc_mc.tight_layout()
            mlflow.log_figure(fig_sc_mc, "fcod_sc_mc_grid.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_sc_mc)

            # Uncertainty zones: alcohol
            fig_zones, ax = plt.subplots(figsize=(12, 5))
            plot_uncertainty_zones(
                fcod_results_ci["alcohol"], ax=ax,
                safe_threshold=0.7, uncertain_threshold=0.3,
                title=f"Decision Zones: Alcohol (α={alpha})",
            )
            mlflow.log_figure(fig_zones, "uncertainty_zones_alcohol.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_zones)

            # Categorical outcome distributions
            _n_cat = len(cat_features)
            _n_cat_rows = math.ceil(_n_cat / N_CAT_COLS)
            fig_cat, axes_cat = plt.subplots(_n_cat_rows, N_CAT_COLS, figsize=(N_CAT_COLS * 6, _n_cat_rows * 5))
            axes_cat_flat = [axes_cat] if _n_cat == 1 else axes_cat.flatten()
            for i in range(_n_cat, _n_cat_rows * N_CAT_COLS):
                axes_cat_flat[i].set_visible(False)
            for idx, (feature_name, config) in enumerate(cat_features.items()):
                ax = axes_cat_flat[idx]
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
                _cls_name = class_names[_cls]
                _prefix = f"by_class/class_{_cls_name}"
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
                    plot_fcod(fcod, ax=_axes.flatten()[idx], outcomes=OUTCOMES, show_ci=True,
                              xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_ci.suptitle(f"Outcome FCODs — Class {_cls_name} (α={alpha})", fontsize=14, y=1.02)
                _fig_ci.tight_layout()
                mlflow.log_figure(_fig_ci, f"{_prefix}/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_ci)

                # Per-feature histogram
                for _feat, _fcod in _cls_fcod_ci.items():
                    _main_ax, _density_ax = plot_fcod(
                        _fcod,
                        outcomes=OUTCOMES,
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
                    plot_stacked_fcod(fcod, ax=_axes.flatten()[idx], outcomes=OUTCOMES,
                                      xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_stacked.suptitle(f"Outcome Distribution — Class {_cls_name} (α={alpha})", fontsize=14, y=1.02)
                _fig_stacked.tight_layout()
                mlflow.log_figure(_fig_stacked, f"{_prefix}/fcod_stacked.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_stacked)

                # SC/MC grid
                _fig_sc_mc = plot_multi_feature_fcod(
                    _cls_fcod_ci, outcomes=["SC", "MC"], n_cols=4,
                    show_ci=True, figsize_per_plot=(6, 4),
                )
                _fig_sc_mc.suptitle(f"SC and MC Rates — Class {_cls_name} (α={alpha})", fontsize=14, y=1.02)
                _fig_sc_mc.tight_layout()
                mlflow.log_figure(_fig_sc_mc, f"{_prefix}/fcod_sc_mc_grid.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_sc_mc)

                # Uncertainty zones: alcohol
                _fig_zones, _ax = plt.subplots(figsize=(12, 5))
                plot_uncertainty_zones(
                    _cls_fcod_ci["alcohol"], ax=_ax,
                    safe_threshold=0.7, uncertain_threshold=0.3,
                    title=f"Decision Zones: Alcohol — Class {_cls_name} (α={alpha})",
                )
                mlflow.log_figure(_fig_zones, f"{_prefix}/uncertainty_zones_alcohol.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_zones)

                # Categorical (single feature for wine)
                _fig_cat, _ax_cat = plt.subplots(1, 1, figsize=(6, 5))
                for feature_name, config in cat_features.items():
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
                    f"Outcome Distribution by Category — Class {_cls_name} (α={alpha})",
                    fontsize=14, y=1.02,
                )
                _fig_cat.tight_layout()
                mlflow.log_figure(_fig_cat, f"{_prefix}/outcome_by_category.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_cat)

            # === Stacked FCOD by class (per feature) ===
            _fv_by_class = {}
            _y_arr = np.asarray(y_test)
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
                    class_names={c: class_names[c] for c in unique_classes},
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

            # === Stacked bars by class — vertical layout (one row per class) ===
            for _feat in _features:
                _fcod_feat = {c: fcod_by_class[c][_feat] for c in unique_classes}
                _fv_feat = {c: _fv_by_class[c][_feat] for c in unique_classes}
                _label = fcod_by_class[unique_classes[0]][_feat].get("feature_name", _feat)
                _fig_bars = plot_stacked_bars_by_class(
                    _fcod_feat,
                    _fv_feat,
                    outcomes=OUTCOMES,
                    class_names={c: class_names[c] for c in unique_classes},
                    xlabel=_label,
                    title=f"{_label} (α={alpha})",
                    alpha=alpha,
                )
                mlflow.log_figure(
                    _fig_bars,
                    f"stacked_bars_by_class/{_feat}.png",
                    save_kwargs={"bbox_inches": "tight"},
                )
                plt.close(_fig_bars)

            # === Merged uncertain outcomes (TS0+TS1 → TS; no-op for multiclass) ===

            # Merged FCOD line plots with CI
            _n_feat_m = len(fcod_results_ci_merged)
            _n_rows_m = math.ceil(_n_feat_m / N_COLS)
            fig_ci_m, axes_m = plt.subplots(_n_rows_m, N_COLS, figsize=(N_COLS * 5, _n_rows_m * 4))
            for i in range(_n_feat_m, _n_rows_m * N_COLS):
                axes_m.flatten()[i].set_visible(False)
            for idx, (feature_name, fcod) in enumerate(fcod_results_ci_merged.items()):
                plot_fcod(fcod, ax=axes_m.flatten()[idx], show_ci=True,
                          xlabel=fcod["feature_name"], title=fcod["feature_name"])
            fig_ci_m.suptitle(f"Outcome FCODs by Feature — Merged TS (α={alpha})", fontsize=14, y=1.02)
            fig_ci_m.tight_layout()
            mlflow.log_figure(fig_ci_m, "merged/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(fig_ci_m)

            # Merged stacked area plots (grid)
            fig_stacked_m, axes_m = plt.subplots(_n_rows_m, N_COLS, figsize=(N_COLS * 5, _n_rows_m * 4))
            for i in range(_n_feat_m, _n_rows_m * N_COLS):
                axes_m.flatten()[i].set_visible(False)
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
            for _cls in unique_classes:
                _cls_name = class_names[_cls]
                _prefix_m = f"merged/by_class/class_{_cls_name}"
                _cls_fcod_ci_m = fcod_ci_by_class_merged[_cls]
                _cls_fcod_m = fcod_by_class_merged[_cls]

                _y_arr = np.asarray(y_test)
                _cls_mask = _y_arr == _cls
                _X_cls = X_test[_cls_mask]

                # Merged FCOD with CI
                _n_feat_cls_m = len(_cls_fcod_ci_m)
                _n_rows_cls_m = math.ceil(_n_feat_cls_m / N_COLS)
                _fig_ci_m2, _axes_m2 = plt.subplots(_n_rows_cls_m, N_COLS, figsize=(N_COLS * 5, _n_rows_cls_m * 4))
                for i in range(_n_feat_cls_m, _n_rows_cls_m * N_COLS):
                    _axes_m2.flatten()[i].set_visible(False)
                for idx, (feat, fcod) in enumerate(_cls_fcod_ci_m.items()):
                    plot_fcod(fcod, ax=_axes_m2.flatten()[idx], show_ci=True,
                              xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_ci_m2.suptitle(f"Outcome FCODs — Class {_cls_name}, Merged TS (α={alpha})", fontsize=14, y=1.02)
                _fig_ci_m2.tight_layout()
                mlflow.log_figure(_fig_ci_m2, f"{_prefix_m}/fcod_with_ci.png", save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_ci_m2)

                # Merged stacked grid per class
                _fig_stacked_m2, _axes_m2 = plt.subplots(_n_rows_cls_m, N_COLS, figsize=(N_COLS * 5, _n_rows_cls_m * 4))
                for i in range(_n_feat_cls_m, _n_rows_cls_m * N_COLS):
                    _axes_m2.flatten()[i].set_visible(False)
                for idx, (feat, fcod) in enumerate(_cls_fcod_m.items()):
                    plot_stacked_fcod(fcod, ax=_axes_m2.flatten()[idx],
                                      xlabel=fcod["feature_name"], title=fcod["feature_name"])
                _fig_stacked_m2.suptitle(f"Outcome Distribution — Class {_cls_name}, Merged TS (α={alpha})", fontsize=14, y=1.02)
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
