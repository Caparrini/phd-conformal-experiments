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
    from conformalpy.explainability import derive_margin_shap, derive_confidence_shap
    from conformalpy.shap import (
        plot_shap_dependence_numeric,
        plot_shap_dependence_categorical,
        make_dependence_grid,
        plot_signed_importance_by_class,
        plot_signed_importance_heatmap,
    )
    from dslib.mlflow_config import load_mlflow_config
    from dslib.data_utils import stratified_split
    from dslib.shap_cache import compute_or_load_shap
    from dslib.tracking import tracked_run

    return (
        ConformalClassifier,
        OmegaConf,
        Path,
        compose,
        compute_or_load_shap,
        derive_confidence_shap,
        derive_margin_shap,
        initialize_config_dir,
        lac_nonconformity,
        load_mlflow_config,
        make_dependence_grid,
        mlflow,
        mo,
        np,
        pd,
        plot_shap_dependence_categorical,
        plot_shap_dependence_numeric,
        plot_signed_importance_by_class,
        plot_signed_importance_heatmap,
        plt,
        stratified_split,
        tracked_run,
    )


@app.cell
def _(Path, compose, initialize_config_dir, load_mlflow_config):
    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="adult")

    mlflow_config = load_mlflow_config()
    mlflow_config.validate_and_log()
    return EXPERIMENT_DIR, cfg, mlflow_config


@app.cell
def _(EXPERIMENT_DIR, cfg, mo, stratified_split):
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

    mo.md(f"**Datos** — Calibración: {len(X_cal):,} | Test: {len(X_test):,}")
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
    model_run_id = _runs.iloc[0].run_id
    pipeline = mlflow.sklearn.load_model(f"runs:/{model_run_id}/model")

    mo.md(f"**Modelo** — run `{model_run_id[:8]}` (`{cfg.experiment.run_name}`)")
    return model_run_id, pipeline


@app.cell
def _(ConformalClassifier, X_cal, cfg, lac_nonconformity, mo, pipeline, y_cal):
    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=cfg.conformal.alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True,
    )
    conf_clf.calibrate(X_cal, y_cal)

    mo.md(f"**Conformal** — α={cfg.conformal.alpha} | Mondrian=True | n_cal={len(X_cal):,}")
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

    # Matriz completa con categóricas como enteros
    X_full = np.column_stack([
        X_test[numeric_cols].values.astype(float),
        np.column_stack([
            X_test[col].map(cat_to_int[col]).values for col in categorical_cols
        ]),
    ])

    # Muestreo background / explain sin solapamiento
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
    y_explain = y_test.values[explain_idx]

    # Callable: decodifica enteros → DataFrame original y devuelve p-values
    def predict_p_values(X: np.ndarray) -> np.ndarray:
        data = {col: X[:, i].astype(float) for i, col in enumerate(numeric_cols)}
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(X[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        return conf_clf.predict_p_values(pd.DataFrame(data, columns=all_cols))

    n_classes = predict_p_values(X_background[:1]).shape[1]
    return (
        X_background,
        X_explain,
        X_explain_df,
        all_cols,
        cat_categories,
        categorical_cols,
        int_to_cat,
        n_classes,
        n_explain,
        numeric_cols,
        predict_p_values,
        y_explain,
    )


@app.cell
def _(
    EXPERIMENT_DIR,
    X_background,
    X_explain,
    cfg,
    compute_or_load_shap,
    mo,
    model_run_id,
    predict_p_values,
):
    # shape: (n_explain, n_features, n_classes)
    shap_values = compute_or_load_shap(
        predict_p_values, X_background, X_explain,
        nsamples=int(cfg.shap.nsamples),
        model_id=model_run_id,
        cache_dir=EXPERIMENT_DIR / "cache",
    )

    mo.md(f"**KernelSHAP** — forma: `{shap_values.shape}` (muestras × features × clases)")
    return (shap_values,)


@app.cell
def _(
    X_explain,
    derive_confidence_shap,
    derive_margin_shap,
    n_classes,
    n_explain,
    np,
    predict_p_values,
    shap_values,
    y_explain,
):
    shap_per_class = [shap_values[:, :, k] for k in range(n_classes)]
    p_values = predict_p_values(X_explain)

    # Margin: SHAP(p_true) - SHAP(p_false) — binario
    shap_p_true = np.array([shap_per_class[int(y_explain[i])][i] for i in range(n_explain)])
    shap_p_false = np.array([shap_per_class[1 - int(y_explain[i])][i] for i in range(n_explain)])
    shap_margin = derive_margin_shap(shap_p_true, shap_p_false)

    # Confidence y credibility
    max_class = np.argmax(p_values, axis=1)
    shap_credibility = np.array([shap_per_class[max_class[i]][i] for i in range(n_explain)])
    shap_confidence = derive_confidence_shap(shap_per_class, p_values)

    # Importancia media |SHAP| por feature (se usa en múltiples celdas downstream)
    shap_derived = {
        **{f"pval_{k}": shap_per_class[k] for k in range(n_classes)},
        "margin": shap_margin,
        "confidence": shap_confidence,
        "credibility": shap_credibility,
    }
    importance = {
        name: np.abs(vals).mean(axis=0) for name, vals in shap_derived.items()
    }
    return importance, shap_derived


@app.cell
def _(mo, n_classes, shap_derived):
    metric_options = list(shap_derived.keys())
    class_options = ["all"] + [str(k) for k in range(n_classes)]

    ui_metric = mo.ui.dropdown(
        options=metric_options,
        value="margin",
        label="Métrica SHAP",
    )
    ui_class = mo.ui.dropdown(
        options=class_options,
        value="all",
        label="Clase verdadera",
    )
    ui_plot_type = mo.ui.dropdown(
        options=["beeswarm", "dependence_numeric", "dependence_categorical",
                 "signed_importance", "heatmap_per_class"],
        value="beeswarm",
        label="Tipo de plot",
    )

    mo.hstack([ui_metric, ui_class, ui_plot_type], justify="start")
    return ui_class, ui_metric, ui_plot_type


@app.cell
def _(
    X_explain_df,
    all_cols,
    cat_categories,
    categorical_cols,
    importance,
    int_to_cat,
    make_dependence_grid,
    mo,
    np,
    numeric_cols,
    plot_shap_beeswarm,
    plot_shap_dependence_categorical,
    plot_shap_dependence_numeric,
    plot_signed_importance_by_class,
    plot_signed_importance_heatmap,
    shap_derived,
    ui_class,
    ui_metric,
    ui_plot_type,
    y_explain,
):
    _metric = ui_metric.value
    _class_filter = ui_class.value
    _plot_type = ui_plot_type.value
    _class_names = {0: "<=50K", 1: ">50K"}

    _shap_vals = shap_derived[_metric]

    # Filtro por clase verdadera
    if _class_filter == "all":
        _mask = np.ones(len(y_explain), dtype=bool)
        _title_suffix = "todas las clases"
    else:
        _k = int(_class_filter)
        _mask = y_explain == _k
        _title_suffix = f"true={_class_names.get(_k, _k)}"

    _shap_filtered = _shap_vals[_mask]
    _X_filtered = X_explain_df[_mask].reset_index(drop=True)
    _y_filtered = y_explain[_mask]

    # ---- Generación del plot según tipo ----
    if _plot_type == "beeswarm":
        fig = plot_shap_beeswarm(
            _shap_filtered, _X_filtered,
            feature_names=all_cols,
            title=f"Beeswarm — {_metric} | {_title_suffix}",
        )

    elif _plot_type == "dependence_numeric":
        _imp = importance[_metric]
        _top_idx = np.argsort(-_imp[:len(numeric_cols)])[:4]
        _top_cols = [numeric_cols[i] for i in _top_idx]
        fig, _axes_pairs = make_dependence_grid(len(_top_cols), ncols=2, figsize=(12, 10))
        for _i, _col in enumerate(_top_cols):
            _col_i = numeric_cols.index(_col)
            plot_shap_dependence_numeric(
                _axes_pairs[_i][0], _X_filtered[_col].values,
                _shap_filtered[:, _col_i], _y_filtered,
                _col, shap_label=_metric, proportion_ax=_axes_pairs[_i][1],
            )
        fig.suptitle(f"Dependencia numérica — {_metric} | {_title_suffix}", fontsize=12)
        fig.tight_layout()

    elif _plot_type == "dependence_categorical":
        _n_num = len(numeric_cols)
        _imp = importance[_metric]
        _top_idx = np.argsort(-_imp[_n_num:])[:4]
        _top_cols = [categorical_cols[j] for j in _top_idx]
        _top_glob = [_n_num + j for j in _top_idx]
        _ncols = min(2, len(_top_cols))
        _nrows = max(1, (len(_top_cols) + 1) // 2)
        fig, _axes_pairs = make_dependence_grid(
            len(_top_cols), ncols=_ncols, figsize=(7 * _ncols, 5 * _nrows)
        )
        for _jj, (_col, _feat_g) in enumerate(zip(_top_cols, _top_glob)):
            _n_cats = len(cat_categories[_col])
            _labels = [int_to_cat[_col][c] for c in range(_n_cats)]
            plot_shap_dependence_categorical(
                _axes_pairs[_jj][0], _X_filtered[_col].values.astype(int),
                _shap_filtered[:, _feat_g], _y_filtered,
                _labels, _col, shap_label=_metric,
                proportion_ax=_axes_pairs[_jj][1],
            )
        fig.suptitle(f"Dependencia categórica — {_metric} | {_title_suffix}", fontsize=12)
        fig.tight_layout()

    elif _plot_type == "signed_importance":
        fig = plot_signed_importance_by_class(
            _shap_filtered, _y_filtered, all_cols,
            class_names=_class_names,
            title=f"Importancia firmada — {_metric} | {_title_suffix}",
        )

    elif _plot_type == "heatmap_per_class":
        fig = plot_signed_importance_heatmap(
            _shap_filtered, _y_filtered, all_cols,
            class_names=_class_names,
            title=f"Heatmap por clase — {_metric} | {_title_suffix}",
        )

    mo.mpl.interactive(fig)
    return


@app.cell
def _(
    all_cols,
    importance,
    mo,
    n_classes,
    np,
    numeric_cols,
    plt,
    shap_derived,
):
    import seaborn as sns  # noqa: F401 — necesario aquí

    # Bar chart: importancia media del margin
    _imp_margin = importance["margin"]
    _sorted_idx = np.argsort(-_imp_margin)
    _colors = [
        "steelblue" if all_cols[i] in numeric_cols else "darkorange"
        for i in _sorted_idx
    ]
    from matplotlib.patches import Patch

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    ax_bar.barh(
        [all_cols[i] for i in _sorted_idx[::-1]],
        _imp_margin[_sorted_idx[::-1]],
        color=_colors[::-1],
    )
    ax_bar.set_xlabel("Mean |SHAP(margin)|")
    ax_bar.set_title("Importancia global — Margin")
    ax_bar.legend(handles=[
        Patch(color="steelblue", label="Numérica"),
        Patch(color="darkorange", label="Categórica"),
    ], loc="lower right")
    fig_bar.tight_layout()

    # Heatmap: importancia normalizada por métrica
    _metric_names = [f"P-val {k}" for k in range(n_classes)] + ["Margin", "Confidence", "Credibility"]
    _heatmap_data = np.column_stack([importance[k] for k in shap_derived])
    _heatmap_norm = _heatmap_data / (_heatmap_data.sum(axis=0) + 1e-10)

    fig_heat, ax_heat = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        _heatmap_norm, ax=ax_heat,
        xticklabels=_metric_names, yticklabels=all_cols,
        annot=True, fmt=".2f", cmap="YlOrRd",
    )
    ax_heat.set_title("Importancia normalizada por métrica SHAP")
    ax_heat.set_xlabel("Métrica")
    ax_heat.set_ylabel("Feature")
    fig_heat.tight_layout()

    mo.vstack([
        mo.md("### Importancia global"),
        mo.hstack([mo.mpl.interactive(fig_bar), mo.mpl.interactive(fig_heat)]),
    ])
    return fig_bar, fig_heat


@app.cell
def _(mo):
    log_button = mo.ui.run_button(label="Registrar en MLflow")
    mo.vstack([mo.md("### Logging MLflow"), log_button])
    return (log_button,)


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
    fig_bar,
    fig_heat,
    importance,
    int_to_cat,
    log_button,
    make_dependence_grid,
    mlflow,
    mlflow_config,
    mo,
    n_classes,
    np,
    numeric_cols,
    plot_shap_dependence_categorical,
    plot_shap_dependence_numeric,
    plot_signed_importance_by_class,
    plot_signed_importance_heatmap,
    plt,
    shap_derived,
    tracked_run,
    y_explain,
):
    if not log_button.value:
        mo.stop(True, mo.md("Pulsa el botón para registrar."))

    _class_names = {0: "<=50K", 1: ">50K"}
    _config_dict = OmegaConf.to_container(cfg, resolve=True)
    _n_num = len(numeric_cols)

    with tracked_run(_config_dict, data_path, cfg.experiment.name,
                     run_name="adult-shap-analysis", mlflow_config=mlflow_config):

        mlflow.log_params({
            "n_background": len(X_background),
            "n_explain": len(X_explain),
            "nsamples_shap": int(cfg.shap.nsamples),
            "n_features_total": len(all_cols),
            "n_features_numeric": len(numeric_cols),
            "n_features_categorical": len(categorical_cols),
        })

        # Importancia global
        for _name, _fig in [("margin_bar", fig_bar), ("normalized_heatmap", fig_heat)]:
            mlflow.log_figure(_fig, f"shap/importance/{_name}.png",
                              save_kwargs={"bbox_inches": "tight"})

        # Beeswarm, signed importance, heatmap por clase — todas las métricas × clases
        for _metric, _shap_vals in shap_derived.items():
            for _class_filter in ["all"] + list(range(n_classes)):
                _mask = (
                    np.ones(len(y_explain), dtype=bool)
                    if _class_filter == "all"
                    else y_explain == _class_filter
                )
                _suffix = "all" if _class_filter == "all" else f"class{_class_filter}"
                _X_sub = X_explain_df[_mask].reset_index(drop=True)
                _y_sub = y_explain[_mask]
                _shap_sub = _shap_vals[_mask]

                # _fig_bee = plot_shap_beeswarm(
                #    _shap_sub, _X_sub, feature_names=all_cols,
                #    title=f"Beeswarm — {_metric} | {_suffix}",
                # )
                # mlflow.log_figure(_fig_bee, f"shap/beeswarm/{_metric}/{_suffix}.png",
                #                   save_kwargs={"bbox_inches": "tight"})
                # plt.close(_fig_bee)

                _fig_si = plot_signed_importance_by_class(
                    _shap_sub, _y_sub, all_cols,
                    class_names=_class_names,
                    title=f"Signed importance — {_metric} | {_suffix}",
                )
                mlflow.log_figure(_fig_si, f"shap/signed_importance/{_metric}/{_suffix}.png",
                                  save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_si)

                _fig_hpc = plot_signed_importance_heatmap(
                    _shap_sub, _y_sub, all_cols,
                    class_names=_class_names,
                    title=f"Heatmap por clase — {_metric} | {_suffix}",
                )
                mlflow.log_figure(_fig_hpc, f"shap/heatmap_per_class/{_metric}/{_suffix}.png",
                                  save_kwargs={"bbox_inches": "tight"})
                plt.close(_fig_hpc)

        # Dependencia numérica — top-4 por métrica
        for _metric, _shap_vals in shap_derived.items():
            _top_idx = np.argsort(-importance[_metric][:_n_num])[:4]
            _top_cols = [numeric_cols[i] for i in _top_idx]
            _fig_dn, _axes_pairs = make_dependence_grid(len(_top_cols), ncols=2, figsize=(12, 10))
            for _i, _col in enumerate(_top_cols):
                _col_i = numeric_cols.index(_col)
                plot_shap_dependence_numeric(
                    _axes_pairs[_i][0], X_explain_df[_col].values,
                    _shap_vals[:, _col_i], y_explain,
                    _col, shap_label=_metric, proportion_ax=_axes_pairs[_i][1],
                )
            _fig_dn.suptitle(f"Dependencia numérica — {_metric}", fontsize=12)
            _fig_dn.tight_layout()
            mlflow.log_figure(_fig_dn, f"shap/dependence/numeric/{_metric}.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig_dn)

        # Dependencia categórica — top-4 por métrica
        for _metric, _shap_vals in shap_derived.items():
            _top_idx = np.argsort(-importance[_metric][_n_num:])[:4]
            _top_cols = [categorical_cols[j] for j in _top_idx]
            _top_glob = [_n_num + j for j in _top_idx]
            _ncols = min(2, len(_top_cols))
            _nrows = max(1, (len(_top_cols) + 1) // 2)
            _fig_dc, _axes_pairs = make_dependence_grid(
                len(_top_cols), ncols=_ncols, figsize=(7 * _ncols, 5 * _nrows)
            )
            for _jj, (_col, _feat_g) in enumerate(zip(_top_cols, _top_glob)):
                _n_cats = len(cat_categories[_col])
                _labels = [int_to_cat[_col][c] for c in range(_n_cats)]
                plot_shap_dependence_categorical(
                    _axes_pairs[_jj][0], X_explain_df[_col].values.astype(int),
                    _shap_vals[:, _feat_g], y_explain,
                    _labels, _col, shap_label=_metric,
                    proportion_ax=_axes_pairs[_jj][1],
                )
            _fig_dc.suptitle(f"Dependencia categórica — {_metric}", fontsize=12)
            _fig_dc.tight_layout()
            mlflow.log_figure(_fig_dc, f"shap/dependence/categorical/{_metric}.png",
                              save_kwargs={"bbox_inches": "tight"})
            plt.close(_fig_dc)

    mo.callout(mo.md("**MLflow** — run `adult-shap-analysis` registrado correctamente."), kind="success")
    return


if __name__ == "__main__":
    app.run()
