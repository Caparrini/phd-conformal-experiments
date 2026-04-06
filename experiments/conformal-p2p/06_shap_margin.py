import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import pandas as pd
    import mlflow

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")

    mo.md("""
    # SHAP Margin Analysis — Conformal P2P Lending

    Full SHAP explainability experiment on the conformal P2P lending model.
    All 8 features (4 numeric + 4 categorical) included via integer encoding.
    Results logged to MLflow.
    """)
    return EXPERIMENT_DIR, OmegaConf, cfg, mlflow, mo, np, pd, plt, shap, sns


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
def _(cfg, mlflow, mo):
    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string="name = 'model'",
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        max_results=1,
    )
    model_id = models.iloc[0].model_id
    pipeline = mlflow.sklearn.load_model(f"models:/{model_id}")
    mo.md(f"Loaded model `{model_id[:8]}` from experiment `{cfg.experiment.name}`")
    return model_id, pipeline


@app.cell
def _(X_cal, cfg, mo, pipeline, y_cal):
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity

    alpha = cfg.conformal.alpha

    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)
    mo.md(f"ConformalClassifier calibrated: α={alpha}, Mondrian=True, n_cal={len(X_cal):,}")
    return (conf_clf,)


@app.cell
def _(X_test, cfg, conf_clf, mo, np, pd, y_test):
    numeric_cols = list(cfg.data.features.numerical)      # ['fico_n', 'dti_n', 'loan_amnt', 'revenue']
    categorical_cols = list(cfg.data.features.categorical)  # 4 categorical features
    all_cols = numeric_cols + categorical_cols              # 8 total

    # Build integer encoders from test set categories
    cat_categories = {col: sorted(X_test[col].dropna().unique()) for col in categorical_cols}
    cat_to_int = {col: {v: i for i, v in enumerate(cats)} for col, cats in cat_categories.items()}
    int_to_cat = {col: {i: v for v, i in m.items()} for col, m in cat_to_int.items()}

    np.random.seed(cfg.experiment.seed)
    n_test = len(X_test)
    background_idx = np.random.choice(n_test, size=1000, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(n_test), background_idx), size=2000, replace=False
    )
    y_explain = y_test.values[explain_idx]
    n_explain = len(explain_idx)

    # Encode categorical columns as integers in numpy arrays
    cat_encoded_test = np.column_stack([
        X_test[col].map(cat_to_int[col]).values for col in categorical_cols
    ])
    X_full = np.column_stack([X_test[numeric_cols].values.astype(float), cat_encoded_test])
    X_background = X_full[background_idx]
    X_explain = X_full[explain_idx]

    n_num = len(numeric_cols)

    def full_pvalue_func(X: np.ndarray) -> np.ndarray:
        """Decode integer-encoded categorical features and return p-values."""
        n = X.shape[0]
        data = {}
        for i, col in enumerate(numeric_cols):
            data[col] = X[:, i].astype(float)
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(X[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        df = pd.DataFrame(data, columns=all_cols)
        return conf_clf.predict_p_values(df)

    # DataFrame of explain set for SHAP plot color axis (integer-encoded categoricals)
    X_explain_df = pd.DataFrame(X_explain, columns=all_cols)

    mo.md(f"""
    **SHAP Setup Complete**

    - Background samples: {len(X_background)} | Explain samples: {n_explain}
    - Features: {len(all_cols)} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)
    - Categorical encoding: integer codes, decoded before each pipeline call
    """)
    return (
        X_background,
        X_explain,
        X_explain_df,
        all_cols,
        cat_categories,
        categorical_cols,
        full_pvalue_func,
        int_to_cat,
        n_explain,
        numeric_cols,
        y_explain,
    )


@app.cell
def _(EXPERIMENT_DIR, X_background, X_explain, full_pvalue_func, mo, model_id):
    from dslib.shap_cache import compute_or_load_shap

    shap_kernel = compute_or_load_shap(
        full_pvalue_func, X_background, X_explain,
        nsamples=10000, model_id=model_id,
        cache_dir=EXPERIMENT_DIR / "cache",
    )
    # shape: (n_samples, n_features, n_outputs)
    mo.md(f"KernelSHAP complete. Array shape: `{shap_kernel.shape}`")
    return (shap_kernel,)


@app.cell
def _(
    X_explain,
    all_cols,
    full_pvalue_func,
    mo,
    n_explain,
    np,
    pd,
    shap_kernel,
    y_explain,
):
    from conformalpy.explainability import derive_margin_shap, derive_confidence_shap

    shap_p0 = shap_kernel[:, :, 0]   # (100, 8)
    shap_p1 = shap_kernel[:, :, 1]   # (100, 8)

    # Compute actual p-values for the explain set
    X_explain_decoded_df = pd.DataFrame(X_explain, columns=all_cols)
    p_values_explain = full_pvalue_func(X_explain)

    # Margin: per-sample selection of true/false class SHAP
    shap_p_true  = np.array([shap_kernel[i, :, y_explain[i]]   for i in range(n_explain)])
    shap_p_false = np.array([shap_kernel[i, :, 1 - y_explain[i]] for i in range(n_explain)])
    shap_margin = derive_margin_shap(shap_p_true, shap_p_false)

    # Credibility: SHAP of the argmax-p-value class per sample
    credibility_class = np.argmax(p_values_explain, axis=1)
    shap_credibility = np.where(credibility_class[:, None] == 0, shap_p0, shap_p1)

    # Confidence: -SHAP of the argmin-p-value class per sample
    shap_confidence = derive_confidence_shap([shap_p0, shap_p1], p_values_explain)

    mo.md(f"""
    **Derived SHAP targets** (all shape `(n_explain, n_features)` = `({n_explain}, {shap_p0.shape[1]})`):

    | Target | Formula | Shape |
    |--------|---------|-------|
    | `shap_p0` | SHAP of p-value class 0 | `{shap_p0.shape}` |
    | `shap_p1` | SHAP of p-value class 1 | `{shap_p1.shape}` |
    | `shap_margin` | SHAP(p_true) − SHAP(p_false) | `{shap_margin.shape}` |
    | `shap_credibility` | SHAP of argmax-p class | `{shap_credibility.shape}` |
    | `shap_confidence` | −SHAP of argmin-p class | `{shap_confidence.shape}` |
    """)
    return shap_confidence, shap_credibility, shap_margin, shap_p0, shap_p1


@app.cell
def _(X_explain_df, all_cols, mo, plt, shap, shap_p0, shap_p1, y_explain):
    figs_beeswarm_pv = {}

    def _beeswarm(shap_vals, feature_df, title: str):
        """Create a SHAP beeswarm figure."""
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_vals, feature_df, feature_names=all_cols,
            plot_type="dot", show=False,
        )
        fig = plt.gcf()
        fig.axes[0].set_title(title, pad=8)
        plt.tight_layout()
        return fig

    # Overall beeswarms
    figs_beeswarm_pv["p0_all"] = _beeswarm(shap_p0, X_explain_df, "SHAP Beeswarm — P-value class 0 (all)")
    figs_beeswarm_pv["p1_all"] = _beeswarm(shap_p1, X_explain_df, "SHAP Beeswarm — P-value class 1 (all)")

    # By true class
    for _k, _label in [(0, "approved"), (1, "default")]:
        _mask = y_explain == _k
        figs_beeswarm_pv[f"p0_class{_k}"] = _beeswarm(
            shap_p0[_mask], X_explain_df[_mask].reset_index(drop=True),
            f"SHAP Beeswarm — P-value class 0 | true={_label}",
        )
        figs_beeswarm_pv[f"p1_class{_k}"] = _beeswarm(
            shap_p1[_mask], X_explain_df[_mask].reset_index(drop=True),
            f"SHAP Beeswarm — P-value class 1 | true={_label}",
        )

    mo.md(f"Beeswarm p-values: {len(figs_beeswarm_pv)} figures generated.")
    return (figs_beeswarm_pv,)


@app.cell
def _(
    X_explain_df,
    all_cols,
    mo,
    plt,
    shap,
    shap_confidence,
    shap_credibility,
    shap_margin,
    y_explain,
):
    figs_beeswarm_derived = {}

    def _beeswarm_d(shap_vals, feature_df, title: str):
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_vals, feature_df, feature_names=all_cols,
            plot_type="dot", show=False,
        )
        fig = plt.gcf()
        fig.axes[0].set_title(title, pad=8)
        plt.tight_layout()
        return fig

    # Margin overall + by true class
    figs_beeswarm_derived["margin_all"] = _beeswarm_d(
        shap_margin, X_explain_df, "SHAP Beeswarm — Margin (all)"
    )
    for _k, _label in [(0, "approved"), (1, "default")]:
        _mask = y_explain == _k
        figs_beeswarm_derived[f"margin_class{_k}"] = _beeswarm_d(
            shap_margin[_mask], X_explain_df[_mask].reset_index(drop=True),
            f"SHAP Beeswarm — Margin | true={_label}",
        )

    # Confidence & credibility overall
    figs_beeswarm_derived["confidence_all"] = _beeswarm_d(
        shap_confidence, X_explain_df, "SHAP Beeswarm — Confidence (all)"
    )
    figs_beeswarm_derived["credibility_all"] = _beeswarm_d(
        shap_credibility, X_explain_df, "SHAP Beeswarm — Credibility (all)"
    )

    # Confidence by class — one figure per class
    for _k, _label in [(0, "approved"), (1, "default")]:
        _mask = y_explain == _k
        figs_beeswarm_derived[f"confidence_class{_k}"] = _beeswarm_d(
            shap_confidence[_mask],
            X_explain_df[_mask].reset_index(drop=True),
            f"Confidence SHAP | true={_label}",
        )

    # Credibility by class — one figure per class
    for _k, _label in [(0, "approved"), (1, "default")]:
        _mask = y_explain == _k
        figs_beeswarm_derived[f"credibility_class{_k}"] = _beeswarm_d(
            shap_credibility[_mask],
            X_explain_df[_mask].reset_index(drop=True),
            f"Credibility SHAP | true={_label}",
        )

    mo.md(f"Beeswarm derived metrics: {len(figs_beeswarm_derived)} figures generated.")
    return (figs_beeswarm_derived,)


@app.cell
def _(
    X_explain_df,
    cat_categories,
    categorical_cols,
    int_to_cat,
    mo,
    numeric_cols,
    shap_margin,
    y_explain,
):
    from conformalpy.shap import (
        make_dependence_grid,
        plot_shap_dependence_categorical,
        plot_shap_dependence_numeric,
    )

    # Numeric dependence plots — 2×2 grid, each numeric feature vs SHAP(margin)
    fig_dep_margin_num, _axes_pairs_num = make_dependence_grid(len(numeric_cols), ncols=2, figsize=(12, 10))
    for _i, _col in enumerate(numeric_cols):
        _scatter_ax, _prop_ax = _axes_pairs_num[_i]
        plot_shap_dependence_numeric(
            _scatter_ax, X_explain_df[_col].values, shap_margin[:, _i],
            y_explain, _col, shap_label="margin", proportion_ax=_prop_ax,
        )
    fig_dep_margin_num.suptitle("SHAP(Margin) Dependence — Numeric Features", fontsize=12)
    fig_dep_margin_num.tight_layout()

    # Categorical dependence plots — jittered strip per category, decoded labels
    fig_dep_margin_cat, _axes_pairs_cat = make_dependence_grid(len(categorical_cols), ncols=2, figsize=(14, 10))
    for _j, _col in enumerate(categorical_cols):
        _scatter_ax, _prop_ax = _axes_pairs_cat[_j]
        _n_cats = len(cat_categories[_col])
        _unique_labels = [int_to_cat[_col][c] for c in range(_n_cats)]
        _int_codes = X_explain_df[_col].values.astype(int)
        plot_shap_dependence_categorical(
            _scatter_ax, _int_codes, shap_margin[:, len(numeric_cols) + _j],
            y_explain, _unique_labels, _col, shap_label="margin", proportion_ax=_prop_ax,
        )
    fig_dep_margin_cat.suptitle("SHAP(Margin) Dependence — Categorical Features", fontsize=12)
    fig_dep_margin_cat.tight_layout()

    mo.md("Dependence plots (margin) generated.")
    return (
        fig_dep_margin_cat,
        fig_dep_margin_num,
        make_dependence_grid,
        plot_shap_dependence_categorical,
        plot_shap_dependence_numeric,
    )


@app.cell
def _(
    X_explain_df,
    cat_categories,
    categorical_cols,
    int_to_cat,
    make_dependence_grid,
    mo,
    numeric_cols,
    plot_shap_dependence_categorical,
    plot_shap_dependence_numeric,
    shap_p0,
    shap_p1,
    y_explain,
):
    # Numeric dependence: p-values — 4×2 grid (features × {p0, p1})
    fig_dep_pv_num, _axes_pairs_pv = make_dependence_grid(len(numeric_cols) * 2, ncols=2, figsize=(12, 22))
    for _i, _col in enumerate(numeric_cols):
        for _k, (_shap_pk, _label) in enumerate([(shap_p0, "P-val class 0"), (shap_p1, "P-val class 1")]):
            _scatter_ax, _prop_ax = _axes_pairs_pv[_i * 2 + _k]
            plot_shap_dependence_numeric(
                _scatter_ax, X_explain_df[_col].values, _shap_pk[:, _i],
                y_explain, _col, shap_label=_label, proportion_ax=_prop_ax,
            )
    fig_dep_pv_num.suptitle("SHAP(P-values) Dependence — Numeric Features", fontsize=12)
    fig_dep_pv_num.tight_layout()

    # Categorical dependence: p-values — 4×2 grid (features × {p0, p1})
    fig_dep_pv_cat, _axes_pairs_pv_cat = make_dependence_grid(len(categorical_cols) * 2, ncols=2, figsize=(14, 24))
    for _j, _col in enumerate(categorical_cols):
        _n_cats = len(cat_categories[_col])
        _int_codes = X_explain_df[_col].values.astype(int)
        _unique_labels = [int_to_cat[_col][c] for c in range(_n_cats)]
        _feat_idx = len(numeric_cols) + _j
        for _k, (_shap_pk, _label) in enumerate([(shap_p0, "P-val class 0"), (shap_p1, "P-val class 1")]):
            _scatter_ax, _prop_ax = _axes_pairs_pv_cat[_j * 2 + _k]
            plot_shap_dependence_categorical(
                _scatter_ax, _int_codes, _shap_pk[:, _feat_idx],
                y_explain, _unique_labels, _col, shap_label=_label, proportion_ax=_prop_ax,
            )
    fig_dep_pv_cat.suptitle("SHAP(P-values) Dependence — Categorical Features", fontsize=12)
    fig_dep_pv_cat.tight_layout()

    mo.md("Dependence plots (p-values) generated.")
    return fig_dep_pv_cat, fig_dep_pv_num


@app.cell
def _(
    X_explain_df,
    cat_categories,
    categorical_cols,
    int_to_cat,
    make_dependence_grid,
    mo,
    numeric_cols,
    plot_shap_dependence_categorical,
    plot_shap_dependence_numeric,
    shap_confidence,
    shap_credibility,
    y_explain,
):
    # Numeric dependence: confidence & credibility — 4 features × 2 metrics
    fig_dep_cc_num, _axes_pairs_cc = make_dependence_grid(len(numeric_cols) * 2, ncols=2, figsize=(12, 22))
    for _i, _col in enumerate(numeric_cols):
        for _k, (_shap_cc, _label) in enumerate([(shap_confidence, "Confidence"), (shap_credibility, "Credibility")]):
            _scatter_ax, _prop_ax = _axes_pairs_cc[_i * 2 + _k]
            plot_shap_dependence_numeric(
                _scatter_ax, X_explain_df[_col].values, _shap_cc[:, _i],
                y_explain, _col, shap_label=_label, proportion_ax=_prop_ax,
            )
    fig_dep_cc_num.suptitle("SHAP(Confidence & Credibility) Dependence — Numeric Features", fontsize=12)
    fig_dep_cc_num.tight_layout()

    # Categorical dependence: confidence & credibility
    fig_dep_cc_cat, _axes_pairs_cc_cat = make_dependence_grid(len(categorical_cols) * 2, ncols=2, figsize=(14, 24))
    for _j, _col in enumerate(categorical_cols):
        _n_cats = len(cat_categories[_col])
        _int_codes = X_explain_df[_col].values.astype(int)
        _unique_labels = [int_to_cat[_col][c] for c in range(_n_cats)]
        _feat_idx = len(numeric_cols) + _j
        for _k, (_shap_cc, _label) in enumerate([(shap_confidence, "Confidence"), (shap_credibility, "Credibility")]):
            _scatter_ax, _prop_ax = _axes_pairs_cc_cat[_j * 2 + _k]
            plot_shap_dependence_categorical(
                _scatter_ax, _int_codes, _shap_cc[:, _feat_idx],
                y_explain, _unique_labels, _col, shap_label=_label, proportion_ax=_prop_ax,
            )
    fig_dep_cc_cat.suptitle("SHAP(Confidence & Credibility) Dependence — Categorical Features", fontsize=12)
    fig_dep_cc_cat.tight_layout()

    mo.md("Dependence plots (confidence & credibility) generated.")
    return fig_dep_cc_cat, fig_dep_cc_num


@app.cell
def _(
    all_cols,
    mo,
    np,
    numeric_cols,
    plt,
    shap_confidence,
    shap_credibility,
    shap_margin,
    shap_p0,
    shap_p1,
    sns,
):
    # --- Feature importance bar chart (mean |SHAP| for margin) ---
    importance_margin = np.abs(shap_margin).mean(axis=0)  # (8,)
    sorted_idx = np.argsort(-importance_margin)
    colors = [
        "steelblue" if col in numeric_cols else "darkorange"
        for col in [all_cols[i] for i in sorted_idx]
    ]

    fig_importance, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(
        [all_cols[i] for i in sorted_idx[::-1]],
        importance_margin[sorted_idx[::-1]],
        color=colors[::-1],
    )
    ax_imp.set_xlabel("Mean |SHAP(margin)|")
    ax_imp.set_title("Feature Importance — Margin SHAP")
    from matplotlib.patches import Patch
    ax_imp.legend(
        handles=[
            Patch(color="steelblue", label="Numeric"),
            Patch(color="darkorange", label="Categorical"),
        ],
        loc="lower right",
    )
    plt.tight_layout()

    # --- Comparative heatmap (all 5 targets) ---
    _metrics = [shap_p0, shap_p1, shap_margin, shap_confidence, shap_credibility]
    _metric_names = ["P-val\nclass 0", "P-val\nclass 1", "Margin", "Confidence", "Credibility"]
    _heatmap_data = np.column_stack([np.abs(m).mean(axis=0) for m in _metrics])  # (8, 5)
    _heatmap_norm = _heatmap_data / (_heatmap_data.sum(axis=0) + 1e-10)          # normalize per metric

    fig_heatmap, ax_hm = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        _heatmap_norm,
        ax=ax_hm,
        xticklabels=_metric_names,
        yticklabels=all_cols,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
    )
    ax_hm.set_title("Normalized Feature Importance Across SHAP Targets")
    ax_hm.set_xlabel("Metric")
    ax_hm.set_ylabel("Feature")
    plt.tight_layout()

    mo.md("Feature importance and comparative heatmap generated.")
    return fig_heatmap, fig_importance


@app.cell
def _(all_cols, mo, plt, shap_confidence, shap_credibility, shap_margin, shap_p0, shap_p1, y_explain):
    from conformalpy.shap import plot_signed_importance_by_class, plot_signed_importance_heatmap

    _class_names = {0: "approved", 1: "default"}

    _targets = {
        "pval_0": shap_p0,
        "pval_1": shap_p1,
        "margin": shap_margin,
        "confidence": shap_confidence,
        "credibility": shap_credibility,
    }

    figs_signed_importance = {}
    for _name, _shap_vals in _targets.items():
        figs_signed_importance[_name] = plot_signed_importance_by_class(
            _shap_vals, y_explain, all_cols,
            class_names=_class_names,
            title=f"Signed SHAP Importance — {_name}",
        )

    figs_heatmap_per_class = {}
    for _name, _shap_vals in _targets.items():
        figs_heatmap_per_class[_name] = plot_signed_importance_heatmap(
            _shap_vals, y_explain, all_cols,
            class_names=_class_names,
            title=f"Per-Class SHAP Heatmap — {_name}",
        )

    mo.md(f"Signed importance: {len(figs_signed_importance)} figures | Heatmaps: {len(figs_heatmap_per_class)} figures")
    return figs_heatmap_per_class, figs_signed_importance, plt


@app.cell
def _(
    OmegaConf,
    cfg,
    data_path,
    fig_dep_cc_cat,
    fig_dep_cc_num,
    fig_dep_margin_cat,
    fig_dep_margin_num,
    fig_dep_pv_cat,
    fig_dep_pv_num,
    fig_heatmap,
    fig_importance,
    figs_beeswarm_derived,
    figs_beeswarm_pv,
    figs_heatmap_per_class,
    figs_signed_importance,
    mlflow,
    mo,
    plt,
):
    from dslib.tracking import tracked_run

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="shap-margin-analysis"):
        mlflow.log_params({
            "n_background": 50,
            "n_explain": 100,
            "nsamples_shap": 500,
            "n_features_total": 8,
            "n_features_numeric": 4,
            "n_features_categorical": 4,
        })

        # Beeswarm — p-values
        for _name, _fig in figs_beeswarm_pv.items():
            mlflow.log_figure(_fig, f"shap/beeswarm/pvalues/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        # Beeswarm — derived metrics
        for _name, _fig in figs_beeswarm_derived.items():
            mlflow.log_figure(_fig, f"shap/beeswarm/derived/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        # Dependence — numeric features
        mlflow.log_figure(fig_dep_margin_num, "shap/dependence/numeric/margin.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_margin_num)
        mlflow.log_figure(fig_dep_pv_num, "shap/dependence/numeric/pvalues.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_pv_num)
        mlflow.log_figure(fig_dep_cc_num, "shap/dependence/numeric/confidence_credibility.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_cc_num)

        # Dependence — categorical features
        mlflow.log_figure(fig_dep_margin_cat, "shap/dependence/categorical/margin.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_margin_cat)
        mlflow.log_figure(fig_dep_pv_cat, "shap/dependence/categorical/pvalues.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_pv_cat)
        mlflow.log_figure(fig_dep_cc_cat, "shap/dependence/categorical/confidence_credibility.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_cc_cat)

        # Importance & heatmap
        mlflow.log_figure(fig_importance, "shap/importance/margin_importance.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_importance)
        mlflow.log_figure(fig_heatmap, "shap/importance/heatmap.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_heatmap)

        for _name, _fig in figs_signed_importance.items():
            mlflow.log_figure(_fig, f"shap/importance/signed/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)
        for _name, _fig in figs_heatmap_per_class.items():
            mlflow.log_figure(_fig, f"shap/importance/heatmap-per-class/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

    mo.md("""
    ## MLflow Logging Complete

    Run `shap-margin-analysis` logged to experiment `conformal-p2p`:

    | Artifact group | Contents |
    |---|---|
    | `shap/beeswarm/pvalues/` | 6 beeswarm figures (p0, p1 — all, class0, class1) |
    | `shap/beeswarm/derived/` | 7 beeswarm figures (margin, confidence, credibility) |
    | `shap/dependence/numeric/` | margin, p-values, confidence+credibility |
    | `shap/dependence/categorical/` | margin, p-values, confidence+credibility |
    | `shap/importance/` | bar chart + normalized heatmap |
    | `shap/importance/signed/` | Signed importance by class per target |
    | `shap/importance/heatmap-per-class/` | Per-class heatmap per target |
    """)
    return


if __name__ == "__main__":
    app.run()
