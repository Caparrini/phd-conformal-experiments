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
        cfg = compose(config_name="credit_card_default")

    mo.md("""
# SHAP Margin Analysis — Credit Card Default (conformal-correctness-margin)

SHAP explainability on the conformal classifier for the Credit Card Default dataset.
All 23 features (14 numeric + 9 categorical) included via integer encoding.
Results logged to MLflow as run `credit-card-shap-analysis`.
    """)
    return EXPERIMENT_DIR, OmegaConf, cfg, mlflow, mo, np, pd, plt, shap, sns


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

    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    mo.md(f"Calibration: {len(X_cal):,} | Test: {len(X_test):,}")
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
    _run_id = _runs.iloc[0].run_id
    pipeline = mlflow.sklearn.load_model(f"runs:/{_run_id}/model")

    mo.md(f"Loaded model from run `{_run_id[:8]}` (`{cfg.experiment.run_name}`)")
    return (pipeline,)


@app.cell
def _(X_cal, cfg, mo, pipeline, y_cal):
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity

    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=cfg.conformal.alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)
    mo.md(f"ConformalClassifier calibrated: α={cfg.conformal.alpha}, Mondrian=True, n_cal={len(X_cal):,}")
    return (conf_clf,)


@app.cell
def _(X_test, cfg, conf_clf, mo, np, pd, y_test):
    numeric_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)
    all_cols = numeric_cols + categorical_cols
    n_num = len(numeric_cols)

    cat_categories = {col: sorted(X_test[col].dropna().unique()) for col in categorical_cols}
    cat_to_int = {col: {v: i for i, v in enumerate(cats)} for col, cats in cat_categories.items()}
    int_to_cat = {col: {i: v for v, i in m.items()} for col, m in cat_to_int.items()}

    np.random.seed(cfg.experiment.seed)
    n_test = len(X_test)
    background_idx = np.random.choice(n_test, size=50, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(n_test), background_idx), size=100, replace=False
    )
    y_explain = y_test.values[explain_idx]
    n_explain = len(explain_idx)

    cat_encoded_test = np.column_stack([
        X_test[col].map(cat_to_int[col]).values for col in categorical_cols
    ])
    X_full = np.column_stack([X_test[numeric_cols].values.astype(float), cat_encoded_test])
    X_background = X_full[background_idx]
    X_explain = X_full[explain_idx]

    def full_pvalue_func(X: np.ndarray) -> np.ndarray:
        """Decode integer-encoded categorical features and return p-values."""
        data = {}
        for i, col in enumerate(numeric_cols):
            data[col] = X[:, i].astype(float)
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(X[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        return conf_clf.predict_p_values(pd.DataFrame(data, columns=all_cols))

    n_classes = full_pvalue_func(X_background[:1]).shape[1]
    X_explain_df = pd.DataFrame(X_explain, columns=all_cols)

    mo.md(f"""
**SHAP Setup Complete**

- Background: {len(X_background)} | Explain: {n_explain} | Classes: {n_classes}
- Features: {len(all_cols)} total ({n_num} numeric, {len(categorical_cols)} categorical)
    """)
    return (
        X_background, X_explain, X_explain_df, all_cols,
        cat_categories, cat_to_int, categorical_cols,
        full_pvalue_func, int_to_cat, n_classes, n_explain, n_num,
        numeric_cols, y_explain,
    )


@app.cell
def _(X_background, X_explain, full_pvalue_func, mo, np, shap):
    kernel_explainer = shap.KernelExplainer(full_pvalue_func, X_background)
    shap_kernel = np.array(kernel_explainer.shap_values(X_explain, nsamples=500))
    # shape: (n_samples, n_features, n_classes)
    mo.md(f"KernelSHAP complete. Array shape: `{shap_kernel.shape}`")
    return kernel_explainer, shap_kernel


@app.cell
def _(X_explain, full_pvalue_func, mo, n_classes, n_explain, np, shap_kernel, y_explain):
    from conformalpy.explainability import derive_margin_shap, derive_confidence_shap

    shap_per_class = [shap_kernel[:, :, k] for k in range(n_classes)]
    p_values_explain = full_pvalue_func(X_explain)

    # Binary: margin = p_true - p_false
    shap_p_true = np.array([shap_per_class[int(y_explain[i])][i] for i in range(n_explain)])
    shap_p_false = np.array([shap_per_class[1 - int(y_explain[i])][i] for i in range(n_explain)])
    shap_margin = derive_margin_shap(shap_p_true, shap_p_false)

    max_class = np.argmax(p_values_explain, axis=1)
    shap_credibility = np.array([shap_per_class[max_class[i]][i] for i in range(n_explain)])
    shap_confidence = derive_confidence_shap(shap_per_class, p_values_explain)

    mo.md(f"Derived SHAP targets: margin {shap_margin.shape}, confidence {shap_confidence.shape}, credibility {shap_credibility.shape}")
    return (
        derive_confidence_shap, derive_margin_shap, max_class, p_values_explain,
        shap_confidence, shap_credibility, shap_margin,
        shap_p_false, shap_p_true, shap_per_class,
    )


@app.cell
def _(X_explain_df, all_cols, mo, n_classes, plt, shap, shap_per_class, y_explain):
    figs_beeswarm_pv = {}

    def _beeswarm(shap_vals, feature_df, title: str):
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_vals, feature_df, feature_names=all_cols, plot_type="dot", show=False,
        )
        fig = plt.gcf()
        fig.axes[0].set_title(title, pad=8)
        plt.tight_layout()
        return fig

    for _k in range(n_classes):
        figs_beeswarm_pv[f"p{_k}_all"] = _beeswarm(
            shap_per_class[_k], X_explain_df, f"SHAP Beeswarm — P-value class {_k} (all)"
        )
        for _j in range(n_classes):
            _mask = y_explain == _j
            figs_beeswarm_pv[f"p{_k}_class{_j}"] = _beeswarm(
                shap_per_class[_k][_mask],
                X_explain_df[_mask].reset_index(drop=True),
                f"SHAP Beeswarm — P-value class {_k} | true={_j}",
            )

    mo.md(f"Beeswarm p-values: {len(figs_beeswarm_pv)} figures generated.")
    return (figs_beeswarm_pv,)


@app.cell
def _(
    X_explain_df, all_cols, mo, n_classes, plt, shap,
    shap_confidence, shap_credibility, shap_margin, y_explain,
):
    figs_beeswarm_derived = {}

    def _beeswarm_d(shap_vals, feature_df, title: str):
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_vals, feature_df, feature_names=all_cols, plot_type="dot", show=False,
        )
        fig = plt.gcf()
        fig.axes[0].set_title(title, pad=8)
        plt.tight_layout()
        return fig

    figs_beeswarm_derived["margin_all"] = _beeswarm_d(shap_margin, X_explain_df, "SHAP Beeswarm — Margin (all)")
    figs_beeswarm_derived["confidence_all"] = _beeswarm_d(shap_confidence, X_explain_df, "SHAP Beeswarm — Confidence (all)")
    figs_beeswarm_derived["credibility_all"] = _beeswarm_d(shap_credibility, X_explain_df, "SHAP Beeswarm — Credibility (all)")

    for _k in range(n_classes):
        _mask = y_explain == _k
        _sub = X_explain_df[_mask].reset_index(drop=True)
        figs_beeswarm_derived[f"margin_class{_k}"] = _beeswarm_d(
            shap_margin[_mask], _sub, f"SHAP Beeswarm — Margin | true={_k}"
        )
        figs_beeswarm_derived[f"confidence_class{_k}"] = _beeswarm_d(
            shap_confidence[_mask], _sub, f"SHAP Beeswarm — Confidence | true={_k}"
        )
        figs_beeswarm_derived[f"credibility_class{_k}"] = _beeswarm_d(
            shap_credibility[_mask], _sub, f"SHAP Beeswarm — Credibility | true={_k}"
        )

    mo.md(f"Beeswarm derived: {len(figs_beeswarm_derived)} figures generated.")
    return (figs_beeswarm_derived,)


@app.cell
def _(
    X_explain_df, all_cols, cat_categories, categorical_cols,
    int_to_cat, mo, np, numeric_cols, plt, shap_margin, y_explain,
):
    _importance = np.abs(shap_margin).mean(axis=0)

    # Top-4 numeric by importance (credit_card has 14, so top-4 matters)
    _top_num_local = sorted(range(len(numeric_cols)), key=lambda i: -_importance[i])[:4]
    _top_num_cols = [numeric_cols[i] for i in _top_num_local]

    # Top-4 categorical by importance (or all if fewer than 4)
    _cat_start = len(numeric_cols)
    _top_cat_local = sorted(range(len(categorical_cols)), key=lambda j: -_importance[_cat_start + j])[:min(4, len(categorical_cols))]
    _top_cat_cols = [categorical_cols[j] for j in _top_cat_local]
    _top_cat_glob_idx = [_cat_start + j for j in _top_cat_local]

    # Numeric 2×2 dependence (top-4)
    fig_dep_margin_num, _axes_num = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    _class_colors = {0: "#e74c3c", 1: "#2ecc71"}
    for _i, _col in enumerate(_top_num_cols):
        _ax = _axes_num[_i // 2, _i % 2]
        _col_i = numeric_cols.index(_col)
        _x_vals = X_explain_df[_col].values
        _y_vals = shap_margin[:, _col_i]
        _ax.scatter(_x_vals, _y_vals, color="steelblue", alpha=0.5, s=20, edgecolors="white", linewidths=0.3)
        _ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        # Global OLS line
        _x_range = np.linspace(_x_vals.min(), _x_vals.max(), 200)
        _coeffs = np.polyfit(_x_vals, _y_vals, 1)
        _ax.plot(_x_range, np.polyval(_coeffs, _x_range), color="black", lw=2, label="Global OLS", zorder=5)
        # Per-class OLS lines
        for _cls, _cls_color in _class_colors.items():
            _cls_mask = y_explain == _cls
            if _cls_mask.sum() > 1:
                _cx, _cy = _x_vals[_cls_mask], _y_vals[_cls_mask]
                _c_coeffs = np.polyfit(_cx, _cy, 1)
                _cx_range = np.linspace(_cx.min(), _cx.max(), 200)
                _ax.plot(_cx_range, np.polyval(_c_coeffs, _cx_range), color=_cls_color, lw=1.8, linestyle="--", label=f"Class {_cls}", zorder=5)
        _ax.legend(fontsize=7, loc="best")
        _ax.set_xlabel(_col)
        _ax.set_ylabel(f"SHAP(margin) — {_col}")
        _ax.set_title(_col)
    plt.suptitle("SHAP(Margin) Dependence — Top-4 Numeric Features", fontsize=12, y=1.01)
    plt.tight_layout()

    # Categorical grid (squeeze=False ensures 2D array)
    _n_top_c = len(_top_cat_cols)
    _cat_nrows = max(1, (_n_top_c + 1) // 2)
    _cat_ncols = min(2, _n_top_c)
    fig_dep_margin_cat, _axes_cat = plt.subplots(
        _cat_nrows, _cat_ncols, figsize=(7 * _cat_ncols, 4.5 * _cat_nrows), squeeze=False
    )
    _axes_flat = _axes_cat.flatten()
    for _jj, (_col, _feat_g) in enumerate(zip(_top_cat_cols, _top_cat_glob_idx)):
        _ax = _axes_flat[_jj]
        _n_cats = len(cat_categories[_col])
        _int_codes = X_explain_df[_col].values.astype(int)
        _unique_codes = np.arange(_n_cats)
        _unique_labels = [int_to_cat[_col][c] for c in _unique_codes]
        _jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(_int_codes))
        _ax.scatter(_int_codes + _jitter, shap_margin[:, _feat_g], alpha=0.5, s=15, color="steelblue")
        _ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        _ax.set_xticks(_unique_codes)
        _ax.set_xticklabels(_unique_labels, rotation=45, ha="right", fontsize=7)
        _ax.set_xlabel(_col)
        _ax.set_ylabel(f"SHAP(margin) — {_col}")
        _ax.set_title(_col)
    for _jj in range(_n_top_c, _cat_nrows * _cat_ncols):
        _axes_flat[_jj].set_visible(False)
    plt.suptitle("SHAP(Margin) Dependence — Top Categorical Features", fontsize=12, y=1.01)
    plt.tight_layout()

    mo.md("Dependence plots (margin) generated.")
    return fig_dep_margin_cat, fig_dep_margin_num


@app.cell
def _(
    all_cols, categorical_cols, mo, n_classes, np, numeric_cols, plt,
    shap_confidence, shap_credibility, shap_margin, shap_per_class, sns,
):
    _importance = np.abs(shap_margin).mean(axis=0)
    _sorted_idx = np.argsort(-_importance)
    _colors = [
        "steelblue" if all_cols[i] in numeric_cols else "darkorange"
        for i in _sorted_idx
    ]

    fig_importance, _ax_imp = plt.subplots(figsize=(10, 7))
    _ax_imp.barh(
        [all_cols[i] for i in _sorted_idx[::-1]],
        _importance[_sorted_idx[::-1]],
        color=_colors[::-1],
    )
    _ax_imp.set_xlabel("Mean |SHAP(margin)|")
    _ax_imp.set_title("Feature Importance — Margin SHAP")
    from matplotlib.patches import Patch
    _ax_imp.legend(handles=[
        Patch(color="steelblue", label="Numeric"),
        Patch(color="darkorange", label="Categorical"),
    ], loc="lower right")
    plt.tight_layout()

    _metrics = shap_per_class + [shap_margin, shap_confidence, shap_credibility]
    _metric_names = [f"P-val\nclass {k}" for k in range(n_classes)] + ["Margin", "Confidence", "Credibility"]
    _heatmap_data = np.column_stack([np.abs(m).mean(axis=0) for m in _metrics])
    _heatmap_norm = _heatmap_data / (_heatmap_data.sum(axis=0) + 1e-10)

    fig_heatmap, _ax_hm = plt.subplots(figsize=(9, 10))
    sns.heatmap(
        _heatmap_norm, ax=_ax_hm,
        xticklabels=_metric_names, yticklabels=all_cols,
        annot=True, fmt=".2f", cmap="YlOrRd",
    )
    _ax_hm.set_title("Normalized Feature Importance Across SHAP Targets")
    _ax_hm.set_xlabel("Metric")
    _ax_hm.set_ylabel("Feature")
    plt.tight_layout()

    mo.md("Feature importance and heatmap generated.")
    return fig_heatmap, fig_importance


@app.cell
def _(
    OmegaConf, all_cols, categorical_cols, cfg, data_path,
    fig_dep_margin_cat, fig_dep_margin_num,
    fig_heatmap, fig_importance,
    figs_beeswarm_derived, figs_beeswarm_pv,
    mlflow, mo, numeric_cols, plt,
):
    from dslib.tracking import tracked_run

    _config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(_config_dict, data_path, cfg.experiment.name, run_name="credit-card-shap-analysis"):
        mlflow.log_params({
            "n_background": 50,
            "n_explain": 100,
            "nsamples_shap": 500,
            "n_features_total": len(all_cols),
            "n_features_numeric": len(numeric_cols),
            "n_features_categorical": len(categorical_cols),
        })

        for _name, _fig in figs_beeswarm_pv.items():
            mlflow.log_figure(_fig, f"shap/beeswarm/pvalues/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        for _name, _fig in figs_beeswarm_derived.items():
            mlflow.log_figure(_fig, f"shap/beeswarm/derived/{_name}.png", save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig)

        mlflow.log_figure(fig_dep_margin_num, "shap/dependence/numeric/margin.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_margin_num)
        mlflow.log_figure(fig_dep_margin_cat, "shap/dependence/categorical/margin.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_dep_margin_cat)

        mlflow.log_figure(fig_importance, "shap/importance/margin_importance.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_importance)
        mlflow.log_figure(fig_heatmap, "shap/importance/heatmap.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_heatmap)

    mo.md("""
## MLflow Logging Complete

Run `credit-card-shap-analysis` logged to experiment `conformal-correctness-margin`.

| Artifact group | Contents |
|---|---|
| `shap/beeswarm/pvalues/` | Beeswarm figures per p-value class (all + per true class) |
| `shap/beeswarm/derived/` | Beeswarm for margin, confidence, credibility |
| `shap/dependence/numeric/` | Top-4 numeric dependence plots |
| `shap/dependence/categorical/` | Top-4 categorical dependence plots |
| `shap/importance/` | Bar chart + normalized heatmap |
    """)
    return


if __name__ == "__main__":
    app.run()
