import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    import shap
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity
    from conformalpy.explainability import derive_margin_shap, select_class_shap
    from conformalpy.shap import (
        plot_shap_dependence_numeric,
        plot_shap_dependence_categorical,
        make_dependence_grid,
    )
    from dslib.mlflow_config import load_mlflow_config
    from dslib.data_utils import load_and_split
    from dslib.shap_cache import compute_or_load_shap
    from dslib.tracking import tracked_run

    return (
        ConformalClassifier,
        OmegaConf,
        Path,
        compose,
        compute_or_load_shap,
        derive_margin_shap,
        initialize_config_dir,
        lac_nonconformity,
        load_and_split,
        load_mlflow_config,
        make_dependence_grid,
        mlflow,
        mo,
        np,
        pd,
        plot_shap_dependence_categorical,
        plot_shap_dependence_numeric,
        plt,
        select_class_shap,
        shap,
        tracked_run,
    )


@app.cell
def _(Path, compose, initialize_config_dir, load_mlflow_config):
    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="globafa2026")

    mlflow_config = load_mlflow_config()
    mlflow_config.validate_and_log()
    return EXPERIMENT_DIR, cfg, mlflow_config


@app.cell
def _(EXPERIMENT_DIR, cfg, load_and_split):
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
def _(cfg, mlflow, mo):
    _experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
    _runs = mlflow.search_runs(
        experiment_ids=[_experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{cfg.experiment.run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = _runs.iloc[0].run_id
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    mo.md(f"Modelo cargado desde run `{run_id[:8]}` (`{cfg.experiment.run_name}`)")
    return pipeline, run_id


@app.cell
def _(ConformalClassifier, X_cal, cfg, lac_nonconformity, mo, pipeline, y_cal):
    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=cfg.conformal.alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)
    mo.md(f"Conformal calibrado — α={cfg.conformal.alpha} | Mondrian=True | n_cal={len(X_cal):,}")
    return (conf_clf,)


@app.cell
def _(X_test, cfg, conf_clf, np, pd, y_test):
    numeric_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)
    all_cols = numeric_cols + categorical_cols
    n_num = len(numeric_cols)

    # Mapas de codificación categórica (entero ↔ etiqueta)
    cat_categories = {
        col: sorted(X_test[col].dropna().unique()) for col in categorical_cols
    }
    cat_to_int = {
        col: {v: i for i, v in enumerate(cats)}
        for col, cats in cat_categories.items()
    }
    int_to_cat = {
        col: {i: v for v, i in mapping.items()}
        for col, mapping in cat_to_int.items()
    }

    # Matriz completa: numéricas como float, categóricas como enteros
    X_full = np.column_stack([
        X_test[numeric_cols].values.astype(float),
        np.column_stack([
            X_test[col].map(cat_to_int[col]).values for col in categorical_cols
        ]),
    ])

    # Muestreo sin solapamiento
    np.random.seed(cfg.experiment.seed)
    n_background = int(cfg.shap.n_background)
    n_explain = int(cfg.shap.n_explain)
    n_test = len(X_test)

    background_idx = np.random.choice(n_test, size=n_background, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(n_test), background_idx),
        size=n_explain,
        replace=False,
    )

    X_background = X_full[background_idx]
    X_explain = X_full[explain_idx]
    X_explain_df = pd.DataFrame(X_explain, columns=all_cols)
    y_explain = y_test.values[explain_idx].astype(int)

    def predict_p_values_fn(X: np.ndarray) -> np.ndarray:
        """Wrapper: decodifica enteros → DataFrame original → conf_clf.predict_p_values."""
        data = {col: X[:, i].astype(float) for i, col in enumerate(numeric_cols)}
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(X[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        return conf_clf.predict_p_values(pd.DataFrame(data, columns=all_cols))

    return (
        X_background,
        X_explain,
        X_explain_df,
        all_cols,
        cat_categories,
        categorical_cols,
        int_to_cat,
        n_explain,
        n_num,
        numeric_cols,
        predict_p_values_fn,
        y_explain,
    )


@app.cell
def _(
    EXPERIMENT_DIR,
    X_background,
    X_explain,
    X_explain_df,
    all_cols,
    categorical_cols,
    cfg,
    compute_or_load_shap,
    int_to_cat,
    mo,
    n_explain,
    np,
    numeric_cols,
    pd,
    pipeline,
    predict_p_values_fn,
    run_id,
    shap,
):
    # --- Classic SHAP via TreeExplainer (exact for tree ensembles) ---
    # METHODOLOGY: TreeExplainer is EXACT (Lundberg et al. 2018). Output space:
    # probability via interventional SHAP with X_background (handles dependent
    # features correctly; path-dependent would inflate spurious correlations).
    # Per-dummy SHAP values aggregated by SUM to per-feature: preserves additivity
    # by linearity of Shapley values across coalitions.
    _xgb_model = pipeline.named_steps["classifier"]
    _preprocessor = pipeline.named_steps["preprocessor"]

    # Decode integer-encoded categoricals back to strings for the preprocessor
    _X_explain_decoded = X_explain_df.copy()
    _X_background_decoded = pd.DataFrame(X_background, columns=all_cols)
    for _col in categorical_cols:
        _X_explain_decoded[_col] = _X_explain_decoded[_col].astype(int).map(int_to_cat[_col])
        _X_background_decoded[_col] = _X_background_decoded[_col].astype(int).map(int_to_cat[_col])

    _X_explain_tr = _preprocessor.transform(_X_explain_decoded)
    _X_background_tr = _preprocessor.transform(_X_background_decoded)

    _explainer_tree = shap.TreeExplainer(
        _xgb_model, _X_background_tr,
        model_output="probability", feature_perturbation="interventional",
    )
    # Binary XGBoost probability output → shape (n_explain, n_features_expanded)
    # Represents SHAP(P(class=1)); complementarity: SHAP(P(class=0)) = -SHAP(P(class=1))
    _shap_expanded = _explainer_tree.shap_values(_X_explain_tr)

    # Aggregate one-hot dummies back to per-semantic-feature SHAP
    _output_indices = _preprocessor.output_indices_
    _cat_encoder = _preprocessor.named_transformers_["cat"]
    _agg_map = {}
    _num_start = _output_indices["num"].start
    for _i, _col in enumerate(numeric_cols):
        _agg_map[_col] = [_num_start + _i]
    _cat_start = _output_indices["cat"].start
    _running = _cat_start
    for _col, _cats in zip(categorical_cols, _cat_encoder.categories_):
        _n_dummies = len(_cats)
        _agg_map[_col] = list(range(_running, _running + _n_dummies))
        _running += _n_dummies

    shap_classic = np.zeros((n_explain, len(all_cols), 2))
    for _i, _col in enumerate(all_cols):
        shap_classic[:, _i, 1] = _shap_expanded[:, _agg_map[_col]].sum(axis=1)
        shap_classic[:, _i, 0] = -shap_classic[:, _i, 1]  # exact binary complementarity

    # --- Conformal SHAP via KernelExplainer (approximate, step-function output) ---
    # METHODOLOGY: No tree-based exact method exists because the conformal p-value
    # F_emp(1 - p_y(x)) is a step function in calibration scores — outside tree
    # representation. KernelExplainer with nsamples=1024 covers 4× the 2^8=256
    # unique coalitions for 8 semantic features.
    shap_conformal = compute_or_load_shap(
        predict_p_values_fn, X_background, X_explain,
        nsamples=int(cfg.shap.nsamples),
        model_id=f"{run_id}-conformal",
        cache_dir=EXPERIMENT_DIR / "cache",
    )
    mo.md(
        f"**Classic SHAP** (TreeExplainer): `{shap_classic.shape}` | "
        f"**Conformal SHAP** (KernelExplainer): `{shap_conformal.shape}`"
    )
    return shap_classic, shap_conformal


@app.cell
def _(derive_margin_shap, select_class_shap, shap_conformal, y_explain):
    shap_p0 = shap_conformal[:, :, 0]          # SHAP de p-value clase 0
    shap_p1 = shap_conformal[:, :, 1]          # SHAP de p-value clase 1
    shap_p_margin = shap_p1 - shap_p0           # SHAP de (p_1 - p_0)

    shap_p_true = select_class_shap(shap_conformal, y_explain)       # SHAP de p_true
    shap_p_false = select_class_shap(shap_conformal, 1 - y_explain)  # SHAP de p_false
    shap_margin = derive_margin_shap(shap_p_true, shap_p_false)      # SHAP de (p_true - p_other)
    return shap_margin, shap_p0, shap_p1, shap_p_margin


@app.cell
def _(np, plt):
    import matplotlib.ticker as mticker

    def plot_shap_importance_grouped(
        shap_vals: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
        name: str,
        colors: dict | None = None,
    ) -> plt.Figure:
        """
        Grouped horizontal bar chart: mean(|SHAP|) global + per class.

        Parameters
        ----------
        shap_vals     : (n_samples, n_features)
        feature_names : ordered list matching axis-1 of shap_vals
        y             : class labels aligned with shap_vals rows
        name          : metric name used in title
        colors        : optional dict {label: hex}, falls back to defaults
        """
        _COLORS = {"Global": "#78909c", "Class 0": "#ef5350", "Class 1": "#42a5f5"}
        if colors:
            _COLORS.update(colors)

        arr = np.asarray(shap_vals)
        y   = np.asarray(y)

        importances = {"Global": np.mean(np.abs(arr), axis=0)}
        for cls in sorted(np.unique(y)):
            mask = y == cls
            if mask.sum() == 0:
                continue
            importances[f"Class {cls}"] = np.mean(np.abs(arr[mask]), axis=0)

        order           = np.argsort(importances["Global"])
        sorted_features = [feature_names[i] for i in order]
        n_features      = len(sorted_features)
        n_groups        = len(importances)

        bar_h  = 0.22
        gap    = 0.08
        step   = n_groups * bar_h + gap
        y_base = np.arange(n_features) * step

        fig, ax = plt.subplots(figsize=(10, max(6, n_features * step * 2.2)))
        fig.patch.set_facecolor("#f8f9fa")
        ax.set_facecolor("#f8f9fa")

        for i, (label, imp) in enumerate(importances.items()):
            sorted_imp = imp[order]
            offset = (i - (n_groups - 1) / 2) * bar_h
            ax.barh(
                y_base + offset, sorted_imp,
                height=bar_h * 0.92,
                label=label,
                color=_COLORS.get(label, f"C{i}"),
                alpha=0.90, edgecolor="#f8f9fa", linewidth=0.5, zorder=3,
            )
            x_max = max(imp.max() for imp in importances.values())
            for j, v in enumerate(sorted_imp):
                ax.text(v + x_max * 0.01, y_base[j] + offset,
                        f"{v:.3f}", va="center", ha="left",
                        fontsize=7.5, color="#616161")

        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.45, color="#9e9e9e", zorder=0)
        ax.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.25, color="#9e9e9e", zorder=0)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bdbdbd")
        ax.tick_params(which="both", length=0)

        ax.set_yticks(y_base)
        ax.set_yticklabels(sorted_features, fontsize=10.5, color="#212121")
        ax.set_xlabel("mean(|SHAP value|)", fontsize=11, color="#424242", labelpad=8)
        ax.set_title(f"SHAP Feature Importance — {name}",
                     fontsize=13, fontweight="bold", color="#212121", pad=16, loc="left")
        ax.text(0, 1.01, "Global importance vs. per-class breakdown",
                transform=ax.transAxes, fontsize=10, color="#757575")
        ax.legend(title="Subset", fontsize=10, title_fontsize=9.5,
                  framealpha=0.95, edgecolor="#e0e0e0", loc="lower right")

        fig.tight_layout(pad=1.4)
        return fig

    return (plot_shap_importance_grouped,)


@app.cell
def _(
    OmegaConf,
    X_background,
    X_explain,
    X_explain_df,
    all_cols,
    cat_categories,
    categorical_cols,
    cfg,
    data_path,
    int_to_cat,
    make_dependence_grid,
    mlflow,
    mlflow_config,
    n_num,
    np,
    numeric_cols,
    plot_shap_dependence_categorical,
    plot_shap_dependence_numeric,
    plot_shap_importance_grouped,
    plt,
    shap,
    shap_classic,
    shap_margin,
    shap_p0,
    shap_p1,
    shap_p_margin,
    tracked_run,
    y_explain,
):
    _config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(
        _config_dict, data_path, cfg.experiment.name,
        run_name="globafa2026-shap-analysis",
        mlflow_config=mlflow_config,
    ):
        mlflow.log_params({
            "n_background": len(X_background),
            "n_explain": len(X_explain),
            "nsamples_shap": int(cfg.shap.nsamples),
            "n_features_total": len(all_cols),
            "n_features_numeric": len(numeric_cols),
            "n_features_categorical": len(categorical_cols),
            "classic_method": "TreeExplainer",
            "classic_output_space": "probability",
            "classic_feature_perturbation": "interventional",
            "conformal_method": "KernelExplainer",
            "conformal_nsamples": int(cfg.shap.nsamples),
            "categorical_encoding_conformal": "integer_round_decode",
            "categorical_encoding_classic": "one_hot_aggregated",
        })

        # --- A. SHAP clásico (sobre predict_proba) --- #
        for _cls, _cls_name in [(0, "class0"), (1, "class1")]:
            shap.summary_plot(
                shap_classic[:, :, _cls], X_explain_df,
                feature_names=all_cols, show=False,
            )
            _fig = plt.gcf()
            _fig.suptitle(f"SHAP Beeswarm — predict_proba | {_cls_name}", y=1.01)
            mlflow.log_figure(_fig, f"shap/classic/beeswarm/{_cls_name}.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        shap.summary_plot(
            shap_classic[:, :, 1], X_explain_df,
            feature_names=all_cols, plot_type="bar", show=False,
        )
        _fig_cimp = plt.gcf()
        _fig_cimp.suptitle("SHAP Importance — predict_proba clase 1 (Default)", y=1.01)
        mlflow.log_figure(_fig_cimp, "shap/classic/importance.png",
                          save_kwargs={"bbox_inches": "tight"})
        plt.close(_fig_cimp)

        # Clip right-tail outliers for numeric dependence plots (mirrors 04_fcod.py fcod_features).
        SHAP_NUMERIC_CLIP_Q = {"dti_n": 0.99, "loan_amnt": 0.99, "revenue": 0.99}

        def _clip_numeric_for_dependence(col, x_vals, shap_vals, y_vals):
            q = SHAP_NUMERIC_CLIP_Q.get(col)
            if q is None:
                return x_vals, shap_vals, y_vals
            hi = np.quantile(x_vals, q)
            mask = x_vals <= hi
            return x_vals[mask], shap_vals[mask], y_vals[mask]

        # --- A2. Classic SHAP dependence plots (class 1 only — binary complementarity) --- #
        _shap_c1 = shap_classic[:, :, 1]
        _imp_c = np.abs(_shap_c1).mean(axis=0)

        _top_num_idx = np.argsort(-_imp_c[:n_num])[:4]
        _top_num_cols = [numeric_cols[i] for i in _top_num_idx]
        if _top_num_cols:
            _fig_dn, _axes_pairs = make_dependence_grid(
                len(_top_num_cols), ncols=2, figsize=(12, 10)
            )
            for _i, _col in enumerate(_top_num_cols):
                _col_i = numeric_cols.index(_col)
                _x, _s, _y = _clip_numeric_for_dependence(
                    _col, X_explain_df[_col].values, _shap_c1[:, _col_i], y_explain,
                )
                plot_shap_dependence_numeric(
                    _axes_pairs[_i][0], _x, _s, _y,
                    _col, shap_label="P(class=1)", proportion_ax=_axes_pairs[_i][1],
                )
            _fig_dn.suptitle("Dependencia numérica — classic SHAP class 1", fontsize=12)
            _fig_dn.tight_layout()
            mlflow.log_figure(_fig_dn, "shap/classic/dependence/numeric/class1.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig_dn)

        _top_cat_idx = np.argsort(-_imp_c[n_num:])[:4]
        _top_cat_cols = [categorical_cols[j] for j in _top_cat_idx]
        _top_cat_glob = [n_num + j for j in _top_cat_idx]
        if _top_cat_cols:
            _ncols = min(2, len(_top_cat_cols))
            _nrows = max(1, (len(_top_cat_cols) + 1) // 2)
            _fig_dc, _axes_pairs = make_dependence_grid(
                len(_top_cat_cols), ncols=_ncols,
                figsize=(7 * _ncols, 5 * _nrows),
            )
            for _jj, (_col, _feat_g) in enumerate(zip(_top_cat_cols, _top_cat_glob)):
                _n_cats = len(cat_categories[_col])
                _labels = [int_to_cat[_col][c] for c in range(_n_cats)]
                plot_shap_dependence_categorical(
                    _axes_pairs[_jj][0], X_explain_df[_col].values.astype(int),
                    _shap_c1[:, _feat_g], y_explain,
                    _labels, _col, shap_label="P(class=1)",
                    proportion_ax=_axes_pairs[_jj][1],
                )
            _fig_dc.suptitle("Dependencia categórica — classic SHAP class 1", fontsize=12)
            _fig_dc.tight_layout()
            mlflow.log_figure(_fig_dc, "shap/classic/dependence/categorical/class1.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig_dc)

        # --- B. Beeswarms conformales --- #
        for _shap_vals, _name in [
            (shap_p0,       "p0"),
            (shap_p1,       "p1"),
            (shap_margin,   "Correctness Margin"),
            (shap_p_margin, "Evidence Margin"),
        ]:
            shap.summary_plot(
                _shap_vals, X_explain_df,
                feature_names=all_cols, show=False,
            )
            _fig = plt.gcf()
            _fig.suptitle(f"SHAP Beeswarm conformal — {_name}", y=1.01)
            mlflow.log_figure(_fig, f"shap/conformal/beeswarm/{_name}.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        # --- C. Dependence plots: margin y p_margin (top-4 por |SHAP| medio) --- #
        for _shap_vals, _name in [
            (shap_margin,   "Correctness Margin"),
            (shap_p_margin, "Evidence Margin"),
        ]:
            _imp = np.abs(_shap_vals).mean(axis=0)

            # Numéricas
            _top_num_idx = np.argsort(-_imp[:n_num])[:4]
            _top_num_cols = [numeric_cols[i] for i in _top_num_idx]
            if _top_num_cols:
                _fig_dn, _axes_pairs = make_dependence_grid(
                    len(_top_num_cols), ncols=2, figsize=(12, 10)
                )
                for _i, _col in enumerate(_top_num_cols):
                    _col_i = numeric_cols.index(_col)
                    _x, _s, _y = _clip_numeric_for_dependence(
                        _col, X_explain_df[_col].values, _shap_vals[:, _col_i], y_explain,
                    )
                    plot_shap_dependence_numeric(
                        _axes_pairs[_i][0], _x, _s, _y,
                        _col, shap_label=_name, proportion_ax=_axes_pairs[_i][1],
                    )
                _fig_dn.suptitle(f"Dependencia numérica — {_name}", fontsize=12)
                _fig_dn.tight_layout()
                mlflow.log_figure(_fig_dn, f"shap/conformal/dependence/numeric/{_name}.png",
                                  save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_dn)

            # Categóricas
            _top_cat_idx = np.argsort(-_imp[n_num:])[:4]
            _top_cat_cols = [categorical_cols[j] for j in _top_cat_idx]
            _top_cat_glob = [n_num + j for j in _top_cat_idx]
            if _top_cat_cols:
                _ncols = min(2, len(_top_cat_cols))
                _nrows = max(1, (len(_top_cat_cols) + 1) // 2)
                _fig_dc, _axes_pairs = make_dependence_grid(
                    len(_top_cat_cols), ncols=_ncols,
                    figsize=(7 * _ncols, 5 * _nrows),
                )
                for _jj, (_col, _feat_g) in enumerate(zip(_top_cat_cols, _top_cat_glob)):
                    _n_cats = len(cat_categories[_col])
                    _labels = [int_to_cat[_col][c] for c in range(_n_cats)]
                    plot_shap_dependence_categorical(
                        _axes_pairs[_jj][0], X_explain_df[_col].values.astype(int),
                        _shap_vals[:, _feat_g], y_explain,
                        _labels, _col, shap_label=_name,
                        proportion_ax=_axes_pairs[_jj][1],
                    )
                _fig_dc.suptitle(f"Dependencia categórica — {_name}", fontsize=12)
                _fig_dc.tight_layout()
                mlflow.log_figure(_fig_dc, f"shap/conformal/dependence/categorical/{_name}.png",
                                  save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_dc)

        # --- D. Importance bars conformales --- #
        for _shap_vals, _name in [
            (shap_margin,   "Correctness Margin"),
            (shap_p_margin, "Evidence Margin"),
        ]:
            # Global
            shap.summary_plot(
                _shap_vals, X_explain_df,
                feature_names=all_cols, plot_type="bar", show=False,
            )
            _fig = plt.gcf()
            _fig.axes[0].set_xlabel("mean(|SHAP value|)", fontsize=10)
            _fig.suptitle(f"SHAP Importance conformal — {_name}", y=1.01)
            mlflow.log_figure(_fig, f"shap/conformal/importance/{_name}.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

            # Por clase
            _shap_arr = np.array(_shap_vals)
            for cls in sorted(np.unique(y_explain)):
                mask = np.asarray(y_explain) == cls
                if mask.sum() == 0:
                    continue
                shap.summary_plot(
                    _shap_arr[mask],
                    X_explain_df.iloc[np.where(mask)[0]],
                    feature_names=all_cols, plot_type="bar", show=False,
                )
                _fig = plt.gcf()
                _fig.axes[0].set_xlabel("mean(|SHAP value|)", fontsize=10)
                _fig.suptitle(
                    f"SHAP Importance conformal — {_name} | class={cls}", y=1.01
                )
                mlflow.log_figure(
                    _fig,
                    f"shap/conformal/importance/{_name}_class{cls}.png",
                    save_kwargs={"bbox_inches": "tight"},
                )
                plt.close(_fig)

        # --- SHAP importance grouped (global + per class) --- #
        for _shap_vals, _name in [
            (shap_margin,   "Correctness Margin"),
            (shap_p_margin, "Evidence Margin"),
        ]:
            _fig = plot_shap_importance_grouped(
                np.array(_shap_vals), all_cols, y_explain, _name,
            )
            mlflow.log_figure(_fig, f"shap/conformal/importance/{_name}_grouped.png",
                              save_kwargs={"bbox_inches": "tight", "dpi": 150})
            plt.close(_fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
