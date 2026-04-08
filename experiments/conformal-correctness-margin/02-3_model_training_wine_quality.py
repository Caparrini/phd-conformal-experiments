import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell
def _(Path):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="wine_quality")

    OmegaConf.to_yaml(cfg)
    return EXPERIMENT_DIR, OmegaConf, cfg


@app.cell
def _():
    """Initialize centralized MLflow configuration."""
    from dslib.mlflow_config import load_mlflow_config
    
    mlflow_config = load_mlflow_config()
    mlflow_config.validate_and_log()
    
    return mlflow_config


@app.cell
def _(EXPERIMENT_DIR, cfg):
    import polars as pl

    data_path = EXPERIMENT_DIR / cfg.data.file
    df = pl.read_parquet(data_path)
    df.shape
    return (data_path,)


@app.cell
def _(cfg, data_path, mo):
    from dslib.data_utils import stratified_split
    from sklearn.preprocessing import LabelEncoder

    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    splits = stratified_split(
        data_path=data_path,
        target=cfg.data.target,
        features=feature_cols,
        train_size=cfg.data.splits.train_size,
        calibration_size=cfg.data.splits.calibration_size,
        seed=cfg.data.splits.seed,
        include_metadata=True,
    )

    X_train, y_train_raw = splits["train"]
    X_cal, y_cal_raw = splits["calibration"]
    X_test, y_test_raw = splits["test"]
    meta = splits["metadata"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_cal = le.transform(y_cal_raw)
    y_test = le.transform(y_test_raw)

    mo.md(f"""
    ## Splits
    | Split | Rows | % of Total |
    |---|---|---|
    | Train | {meta['train']['n_samples']:,} | {meta['train']['pct_total']:.1%} |
    | Calibration | {meta['calibration']['n_samples']:,} | {meta['calibration']['pct_total']:.1%} |
    | Test | {meta['test']['n_samples']:,} | {meta['test']['pct_total']:.1%} |

    Classes: {list(le.classes_)}
    """)
    return X_cal, X_test, X_train, y_cal, y_test, y_train


@app.cell
def _(cfg):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    numerical_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)

    transformers = [("num", "passthrough", numerical_cols)]
    if categorical_cols:
        from sklearn.preprocessing import OneHotEncoder

        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )
    preprocessor = ColumnTransformer(transformers=transformers)

    model = XGBClassifier(**dict(cfg.model.params), n_jobs=-1, verbosity=0)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return (pipeline,)


@app.cell
def _(
    OmegaConf,
    X_cal,
    X_test,
    X_train,
    cfg,
    data_path,
    mlflow_config,
    mo,
    pipeline,
    y_cal,
    y_test,
    y_train,
):
    import mlflow
    from dslib.tracking import tracked_run
    from sklearn.metrics import (
        roc_auc_score,
        balanced_accuracy_score,
        f1_score,
    )

    def _metrics(prefix, y, proba, pred):
        return {
            f"{prefix}roc_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
            f"{prefix}balanced_accuracy": balanced_accuracy_score(y, pred),
            f"{prefix}f1_macro": f1_score(y, pred, average="macro"),
        }

    mlflow.sklearn.autolog()
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="wine-xgboost", mlflow_config=mlflow_config):
        pipeline.fit(X_train, y_train)

        train_metrics = _metrics(
            "train_", y_train, pipeline.predict_proba(X_train), pipeline.predict(X_train)
        )
        mlflow.log_metrics(train_metrics)

        cal_metrics = _metrics(
            "cal_", y_cal, pipeline.predict_proba(X_cal), pipeline.predict(X_cal)
        )
        mlflow.log_metrics(cal_metrics)

        test_metrics = _metrics(
            "test_", y_test, pipeline.predict_proba(X_test), pipeline.predict(X_test)
        )
        mlflow.log_metrics(test_metrics)

    mo.md(f"""
    ## Training Complete
    | Metric | Train | Calibration | Test |
    |---|---|---|---|
    | **ROC-AUC (macro OvR)** | {train_metrics["train_roc_auc"]:.4f} | {cal_metrics["cal_roc_auc"]:.4f} | {test_metrics["test_roc_auc"]:.4f} |
    | **Balanced Accuracy** | {train_metrics["train_balanced_accuracy"]:.1%} | {cal_metrics["cal_balanced_accuracy"]:.1%} | {test_metrics["test_balanced_accuracy"]:.1%} |
    | **F1 (macro)** | {train_metrics["train_f1_macro"]:.4f} | {cal_metrics["cal_f1_macro"]:.4f} | {test_metrics["test_f1_macro"]:.4f} |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
