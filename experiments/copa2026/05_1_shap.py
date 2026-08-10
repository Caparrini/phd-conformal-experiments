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
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from scipy.stats import spearmanr, pearsonr

    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity
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
        initialize_config_dir,
        lac_nonconformity,
        load_mlflow_config,
        mlflow,
        mo,
        np,
        pd,
        pearsonr,
        plt,
        spearmanr,
        stratified_split,
        tracked_run,
    )


@app.cell
def _(Path, compose, initialize_config_dir, load_mlflow_config):
    # Reutiliza el mismo config que 05_shap.py (copa2026.yaml) — no toca el original.
    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="copa2026")

    mlflow_config = load_mlflow_config()
    mlflow_config.validate_and_log()
    return EXPERIMENT_DIR, cfg, mlflow_config


@app.cell
def _(EXPERIMENT_DIR, cfg, stratified_split):
    # NOTA: asumo que stratified_split devuelve también la partición "train".
    # Si la key difiere en dslib.data_utils, ajustar aquí (única dependencia frágil).
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
    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    return X_cal, X_test, X_train, data_path, y_cal, y_test, y_train


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
    mo.md(f"Modelo cargado desde run `{run_id[:8]}` (`{cfg.experiment.run_name}`) — mismo run que 05_shap.py")
    return pipeline, run_id


@app.cell
def _(ConformalClassifier, X_cal, cfg, lac_nonconformity, mo, pipeline, y_cal):
    # Mismo conformal classifier que en el experimento principal: calibrado SIEMPRE sobre
    # X_cal, independientemente de qué split se use como background de SHAP.
    # El experimento de sensibilidad varía el background, no la calibración.
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
def _(X_cal, X_test, X_train, cfg, conf_clf, np, pd, y_test):
    # Codificación categórica: los mapas cat_to_int/int_to_cat se construyen sobre la UNION
    # de categorías de los tres splits, no solo sobre X_test. Si una categoría rara aparece
    # solo en train o cal y no en test, el encoding original de 05_shap.py la ignoraría al
    # samplear background de esos splits — este fix es necesario para que el experimento
    # de sensibilidad sea válido.
    numeric_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)
    all_cols = numeric_cols + categorical_cols
    n_num = len(numeric_cols)

    cat_categories = {
        col: sorted(
            pd.concat([X_train[col], X_cal[col], X_test[col]]).dropna().unique()
        )
        for col in categorical_cols
    }
    cat_to_int = {
        col: {v: i for i, v in enumerate(cats)} for col, cats in cat_categories.items()
    }
    int_to_cat = {
        col: {i: v for v, i in mapping.items()} for col, mapping in cat_to_int.items()
    }

    def to_encoded_matrix(df):
        return np.column_stack([
            df[numeric_cols].values.astype(float),
            np.column_stack([df[col].map(cat_to_int[col]).values for col in categorical_cols]),
        ])

    X_train_full = to_encoded_matrix(X_train)
    X_cal_full = to_encoded_matrix(X_cal)
    X_test_full = to_encoded_matrix(X_test)

    def predict_p_values_fn(X: np.ndarray) -> np.ndarray:
        data = {col: X[:, i].astype(float) for i, col in enumerate(numeric_cols)}
        for j, col in enumerate(categorical_cols):
            n_cats = len(cat_categories[col])
            codes = np.round(X[:, n_num + j]).astype(int).clip(0, n_cats - 1)
            data[col] = [int_to_cat[col][c] for c in codes]
        return conf_clf.predict_p_values(pd.DataFrame(data, columns=all_cols))

    return (
        X_cal_full,
        X_test_full,
        X_train_full,
        all_cols,
        cat_categories,
        categorical_cols,
        int_to_cat,
        n_num,
        numeric_cols,
        predict_p_values_fn,
        to_encoded_matrix,
    )


@app.cell
def _(X_cal_full, X_test_full, X_train_full, all_cols, cfg, np, pd, y_test):
    # Instancias a explicar: EXACTAMENTE las mismas que en 05_shap.py (misma seed,
    # misma lógica de muestreo sobre X_test), para que la comparación entre backgrounds
    # sea sobre el mismo conjunto explicado y no introduzca una segunda fuente de varianza.
    np.random.seed(cfg.experiment.seed)
    n_background = int(cfg.shap.n_background)  # 300
    n_explain = int(cfg.shap.n_explain)         # 1000
    n_test = len(X_test_full)

    background_idx_test = np.random.choice(n_test, size=n_background, replace=False)
    explain_idx = np.random.choice(
        np.setdiff1d(np.arange(n_test), background_idx_test),
        size=n_explain,
        replace=False,
    )

    X_explain = X_test_full[explain_idx]
    X_explain_df = pd.DataFrame(X_explain, columns=all_cols)
    y_explain = y_test.values[explain_idx].astype(int)

    # Backgrounds de train y cal: mismo tamaño (300), misma seed, sampleados de su propio
    # split. No hace falta disjunción respecto a X_explain porque train/cal ya son
    # disjuntos de test por construcción del split original.
    background_idx_train = np.random.choice(len(X_train_full), size=n_background, replace=False)
    background_idx_cal = np.random.choice(len(X_cal_full), size=n_background, replace=False)

    X_background_test = X_test_full[background_idx_test]
    X_background_train = X_train_full[background_idx_train]
    X_background_cal = X_cal_full[background_idx_cal]

    return (
        X_background_cal,
        X_background_test,
        X_background_train,
        X_explain,
        X_explain_df,
        n_explain,
        y_explain,
    )


@app.cell
def _(
    EXPERIMENT_DIR,
    X_background_cal,
    X_background_test,
    X_background_train,
    X_explain,
    cfg,
    compute_or_load_shap,
    mo,
    predict_p_values_fn,
    run_id,
):
    # nsamples reducido para train/cal: es un diagnóstico de sensibilidad, no el resultado
    # principal — no necesita la misma precisión que el análisis reportado en el paper.
    # 256 = 2^8 coaliciones exactas para 8 features semánticas (cota inferior razonable).
    _nsamples_full = int(cfg.shap.nsamples)      # el mismo que en 05_shap.py, p.ej. 1024
    _nsamples_diag = min(256, _nsamples_full)

    shap_test = compute_or_load_shap(
        predict_p_values_fn, X_background_test, X_explain,
        nsamples=_nsamples_full,
        model_id=f"{run_id}-conformal",              # mismo cache que 05_shap.py: no recalcula
        cache_dir=EXPERIMENT_DIR / "cache",
    )
    shap_train = compute_or_load_shap(
        predict_p_values_fn, X_background_train, X_explain,
        nsamples=_nsamples_diag,
        model_id=f"{run_id}-conformal-bg_train",
        cache_dir=EXPERIMENT_DIR / "cache",
    )
    shap_cal = compute_or_load_shap(
        predict_p_values_fn, X_background_cal, X_explain,
        nsamples=_nsamples_diag,
        model_id=f"{run_id}-conformal-bg_cal",
        cache_dir=EXPERIMENT_DIR / "cache",
    )

    mo.md(
        f"SHAP calculado con tres backgrounds — test: `{shap_test.shape}` (nsamples={_nsamples_full}), "
        f"train: `{shap_train.shape}` (nsamples={_nsamples_diag}), "
        f"cal: `{shap_cal.shape}` (nsamples={_nsamples_diag})"
    )
    return shap_cal, shap_test, shap_train


@app.cell
def _(all_cols, np, pearsonr, shap_cal, shap_test, shap_train, spearmanr):
    # Métrica 1 — Spearman sobre el RANKING de importancia global (mean|shap| por feature):
    # es la que importa para las figuras del paper (bar plots de importancia).
    # Métrica 2 — Pearson sobre los valores φ_j aplanados (instancia × feature):
    # captura si además coincide la magnitud/dirección punto a punto, no solo el orden.

    def shap_p1(shap_arr):
        # p-value de clase 1, mismo objeto que shap_p1 en 05_shap.py
        return np.asarray(shap_arr)[:, :, 1]

    _s_test = shap_p1(shap_test)
    _s_train = shap_p1(shap_train)
    _s_cal = shap_p1(shap_cal)

    imp_test = np.abs(_s_test).mean(axis=0)
    imp_train = np.abs(_s_train).mean(axis=0)
    imp_cal = np.abs(_s_cal).mean(axis=0)

    spearman_train, _ = spearmanr(imp_test, imp_train)
    spearman_cal, _ = spearmanr(imp_test, imp_cal)

    pearson_train, _ = pearsonr(_s_test.ravel(), _s_train.ravel())
    pearson_cal, _ = pearsonr(_s_test.ravel(), _s_cal.ravel())

    results_summary = {
        "spearman_rank_importance_test_vs_train": float(spearman_train),
        "spearman_rank_importance_test_vs_cal": float(spearman_cal),
        "pearson_shap_values_test_vs_train": float(pearson_train),
        "pearson_shap_values_test_vs_cal": float(pearson_cal),
    }

    per_feature_table = {
        "feature": all_cols,
        "importance_test": imp_test,
        "importance_train": imp_train,
        "importance_cal": imp_cal,
    }

    return per_feature_table, results_summary


@app.cell
def _(mo, results_summary):
    mo.md(f"""
    **Resultado E3 — sensibilidad del background SHAP**

    | Comparación | Spearman (ranking importancia) | Pearson (valores φ_j) |
    |---|---|---|
    | test vs train | {results_summary['spearman_rank_importance_test_vs_train']:.4f} | {results_summary['pearson_shap_values_test_vs_train']:.4f} |
    | test vs cal | {results_summary['spearman_rank_importance_test_vs_cal']:.4f} | {results_summary['pearson_shap_values_test_vs_cal']:.4f} |

    Si ambos Spearman > ~0.9, la elección de background es empíricamente inmaterial
    para el ranking de importancia.
    """)
    return


@app.cell
def _(all_cols, np, per_feature_table, plt):
    # Bar plot horizontal comparando importancia por feature entre los tres backgrounds.
    _order = np.argsort(per_feature_table["importance_test"])
    _features = [all_cols[i] for i in _order]
    _bar_h = 0.25

    fig, ax = plt.subplots(figsize=(9, max(5, len(_features) * 0.5)))
    _y = np.arange(len(_features))
    ax.barh(_y - _bar_h, per_feature_table["importance_test"][_order], height=_bar_h, label="test (principal)", color="#42a5f5")
    ax.barh(_y, per_feature_table["importance_train"][_order], height=_bar_h, label="train", color="#ef5350")
    ax.barh(_y + _bar_h, per_feature_table["importance_cal"][_order], height=_bar_h, label="cal", color="#66bb6a")
    ax.set_yticks(_y)
    ax.set_yticklabels(_features)
    ax.set_xlabel("mean(|SHAP value|) — p-value clase 1")
    ax.set_title("E3 — Sensibilidad del background: importancia por feature")
    ax.legend()
    fig.tight_layout()
    return (fig,)


@app.cell
def _(OmegaConf, cfg, data_path, fig, mlflow, mlflow_config, results_summary, tracked_run):
    _config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(
        _config_dict, data_path, cfg.experiment.name,
        run_name="copa2026-shap-background-sensitivity",
        mlflow_config=mlflow_config,
    ):
        mlflow.log_params({
            "experiment": "E3_background_sensitivity",
            "background_train_test": "test (principal, 05_shap.py)",
            "background_alt_1": "train",
            "background_alt_2": "calibration",
        })
        mlflow.log_metrics(results_summary)
        mlflow.log_figure(fig, "shap/sensitivity/background_comparison.png",
                          save_kwargs={"bbox_inches": "tight"})
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()