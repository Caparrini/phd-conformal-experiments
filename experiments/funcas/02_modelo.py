import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import mlflow
    from omegaconf import OmegaConf
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    from dslib.experiment_io import (
        load_cfg, load_splits, make_preprocessor, plot_confusion, split_metrics,
    )
    from dslib.mlflow_config import local_mlflow_config
    from dslib.tracking import tracked_run

    EXPERIMENT_DIR = Path(__file__).parent
    return (
        EXPERIMENT_DIR, OmegaConf, Pipeline, XGBClassifier, compute_sample_weight,
        load_cfg, load_splits, local_mlflow_config, make_preprocessor, mlflow,
        mo, plot_confusion, split_metrics, tracked_run,
    )


@app.cell
def _(EXPERIMENT_DIR, load_cfg, local_mlflow_config):
    cfg = load_cfg(EXPERIMENT_DIR, "funcas")

    # Seguimiento local: base SQLite y artefactos dentro de esta carpeta.
    mlflow_config = local_mlflow_config(EXPERIMENT_DIR)
    mlflow_config.validate_and_log()
    return cfg, mlflow_config


@app.cell
def _(mo):
    mo.md("""
    # Modelo base de tres clases

    Entrenamos un XGBoost que da, para cada préstamo, tres probabilidades: no
    default, dudoso y default. Esas probabilidades son lo único que usa la
    predicción conforme del notebook 03. El modelo no sabe nada de conformal.

    ## 1. Partición

    Separamos los préstamos al azar en tres grupos con la misma proporción de
    clases en cada uno:

    - **Entrenamiento (60%)**: el modelo aprende aquí.
    - **Calibración (20%)**: se reserva para el paso conforme. El modelo no la
      ve al entrenar.
    - **Test (20%)**: se usa solo para medir.

    El azar importa. La garantía del conformal exige que calibración y test
    sean intercambiables: que un préstamo de test se comporte como uno de
    calibración. Mezclando al azar se cumple por construcción. El notebook 03
    (sección 9) muestra qué pasa cuando no se cumple.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, cfg, load_splits, mo):
    splits = load_splits(cfg, EXPERIMENT_DIR)
    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    names = list(cfg.data.class_names)

    _rows = "\n".join(
        f"| {label} | {len(y):,} | "
        + " | ".join(f"{y.value_counts(normalize=True).get(k, 0):.1%}" for k in range(len(names)))
        + " |"
        for label, y in [("Entrenamiento", y_train), ("Calibración", y_cal), ("Test", y_test)]
    )
    mo.md(
        f"""
    | Partición | Préstamos | {" | ".join(names)} |
    |---|---|{"---|" * len(names)}
    {_rows}
    """
    )
    return X_cal, X_test, X_train, names, y_cal, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## 2. Modelo

    Las variables numéricas pasan tal cual y las categóricas a *one-hot*. La
    clase dudosa es mucho menos frecuente que las otras, así que damos más peso
    a las clases raras al entrenar. Esto cambia las probabilidades que da el
    modelo, pero no la validez del conformal, que se calibra sobre las
    probabilidades que salgan.
    """)
    return


@app.cell
def _(Pipeline, XGBClassifier, cfg, make_preprocessor, mo):
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(cfg)),
            ("classifier", XGBClassifier(**dict(cfg.model.params), n_jobs=-1, verbosity=0)),
        ]
    )
    _rows = "\n".join(f"| `{k}` | {v} |" for k, v in dict(cfg.model.params).items())
    mo.md(
        f"""
    Hiperparámetros del XGBoost (vienen de `config/funcas.yaml`):

    | Parámetro | Valor |
    |---|---|
    {_rows}
    """
    )
    return (pipeline,)


@app.cell
def _(compute_sample_weight, mo, names, y_train):
    sample_weights = compute_sample_weight("balanced", y_train)
    _y = y_train.to_numpy()
    _rows = "\n".join(
        f"| {names[k]} | {(_y == k).sum():,} | {(_y == k).mean():.1%} | "
        f"{sample_weights[_y == k][0]:.2f} |"
        for k in range(len(names))
    )
    mo.md(
        f"""
    **Pesos de entrenamiento.** `compute_sample_weight("balanced")` da a cada
    préstamo el peso `n / (3 · n_clase)`, de modo que la suma de pesos es la
    misma en las tres clases.

    | Clase | Préstamos | Proporción | Peso por préstamo |
    |---|---|---|---|
    {_rows}
    """
    )
    return (sample_weights,)


@app.cell
def _(
    EXPERIMENT_DIR, OmegaConf, X_cal, X_test, X_train, cfg, mlflow, mlflow_config,
    names, pipeline, plot_confusion, sample_weights, split_metrics, tracked_run,
    y_cal, y_test, y_train,
):
    mlflow.sklearn.autolog()

    with tracked_run(
        OmegaConf.to_container(cfg, resolve=True),
        EXPERIMENT_DIR / cfg.data.file,
        cfg.experiment.name,
        run_name=cfg.experiment.run_name,
        mlflow_config=mlflow_config,
    ):
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

        metrics = {}
        for _prefix, _X, _y in [
            ("train_", X_train, y_train),
            ("cal_", X_cal, y_cal),
            ("test_", X_test, y_test),
        ]:
            metrics.update(split_metrics(_prefix, _y, pipeline.predict_proba(_X)))
        mlflow.log_metrics(metrics)

        fig_confusion = plot_confusion(
            y_test, pipeline.predict(X_test), names, "XGBoost, test: clase predicha por clase real"
        )
        mlflow.log_figure(fig_confusion, "confusion_matrix_test.png")

    fig_confusion
    return (metrics,)


@app.cell
def _(metrics, mo, names):
    _rows = "\n".join(
        f"| {label} | {metrics[p + 'log_loss']:.3f} | {metrics[p + 'balanced_accuracy']:.3f} | "
        + " | ".join(f"{metrics[f'{p}recall_{k}']:.1%}" for k in range(len(names)))
        + " |"
        for label, p in [("Entrenamiento", "train_"), ("Calibración", "cal_"), ("Test", "test_")]
    )
    mo.md(
        f"""
    ## 3. Resultados

    | Partición | Log-loss | Balanced acc. | {" | ".join(f"Recall {n}" for n in names)} |
    |---|---|---|{"---|" * len(names)}
    {_rows}

    El *recall* de una clase es la fracción de préstamos de esa clase que el
    modelo acierta cuando se queda con la clase más probable: es la diagonal de
    la matriz de confusión de arriba. La *balanced accuracy* es la media de los
    tres recalls.
    """
    )
    return


if __name__ == "__main__":
    app.run()
