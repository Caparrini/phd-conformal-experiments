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
        cfg = compose(config_name="config")

    OmegaConf.to_yaml(cfg)
    return EXPERIMENT_DIR, OmegaConf, cfg


@app.cell
def _(EXPERIMENT_DIR, cfg):
    import polars as pl

    data_path = EXPERIMENT_DIR / cfg.data.file
    df = pl.read_parquet(data_path)
    df.shape
    return (data_path,)


@app.cell
def _(cfg, data_path, mo):
    from dslib.data_utils import load_and_split

    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    splits = load_and_split(
        data_path=data_path,
        date_column=cfg.data.splits.date_column,
        train_cutoff=cfg.data.splits.train_cutoff,
        calibration_cutoff=cfg.data.splits.calibration_cutoff,
        target=cfg.data.target,
        features=feature_cols,
        include_metadata=True,
    )

    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    meta = splits["metadata"]

    mo.md(f"""
    ## Splits
    | Split | Rows | Default Rate | Date From | Date To | % of Total |
    |---|---|---|---|---|---|
    | Train | {meta['train']['n_samples']:,} | {meta['train']['default_rate']:.1%} | {meta['train']['date_from']} | {meta['train']['date_to']} | {meta['train']['pct_total']:.1%} |
    | Calibration | {meta['calibration']['n_samples']:,} | {meta['calibration']['default_rate']:.1%} | {meta['calibration']['date_from']} | {meta['calibration']['date_to']} | {meta['calibration']['pct_total']:.1%} |
    | Test | {meta['test']['n_samples']:,} | {meta['test']['default_rate']:.1%} | {meta['test']['date_from']} | {meta['test']['date_to']} | {meta['test']['pct_total']:.1%} |
    """)
    return X_cal, X_test, X_train, y_cal, y_test, y_train


@app.cell
def _(cfg):
    from sklearn.preprocessing import OneHotEncoder  # , StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    numerical_cols = list(cfg.data.features.numerical)
    categorical_cols = list(cfg.data.features.categorical)

    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", StandardScaler(), numerical_cols), # It is not necessary for XGBoost
            ("num", "passthrough", numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

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
        precision_score,
        recall_score,
    )

    mlflow.autolog()
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(
        config_dict, data_path, cfg.experiment.name, run_name="xgboost-baseline"
    ):
        pipeline.fit(X_train, y_train)

        # Train metrics
        y_pred_train = pipeline.predict(X_train)
        y_proba_train = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = {
            "train_roc_auc": roc_auc_score(y_train, y_proba_train),
            "train_balanced_accuracy": balanced_accuracy_score(y_train, y_pred_train),
            "train_f1_macro": f1_score(y_train, y_pred_train, average="macro"),
            "train_precision_default": precision_score(y_train, y_pred_train),
            "train_recall_default": recall_score(y_train, y_pred_train),
        }
        mlflow.log_metrics(train_metrics)

        # Test metrics
        y_pred_test = pipeline.predict(X_test)
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_proba_test),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
            "test_f1_macro": f1_score(y_test, y_pred_test, average="macro"),
            "test_precision_default": precision_score(y_test, y_pred_test),
            "test_recall_default": recall_score(y_test, y_pred_test),
        }
        mlflow.log_metrics(test_metrics)

        # Calibration metrics
        y_pred_cal = pipeline.predict(X_cal)
        y_proba_cal = pipeline.predict_proba(X_cal)[:, 1]
        cal_metrics = {
            "cal_roc_auc": roc_auc_score(y_cal, y_proba_cal),
            "cal_balanced_accuracy": balanced_accuracy_score(y_cal, y_pred_cal),
            "cal_f1_macro": f1_score(y_cal, y_pred_cal, average="macro"),
            "cal_precision_default": precision_score(y_cal, y_pred_cal),
            "cal_recall_default": recall_score(y_cal, y_pred_cal),
        }
        mlflow.log_metrics(cal_metrics)

    mo.md(f"""
    ## Training Complete
    | Metric | Train | Test | Calibration |
    |---|---|---|---|
    | **ROC-AUC** | {train_metrics["train_roc_auc"]:.4f} | {test_metrics["test_roc_auc"]:.4f} | {cal_metrics["cal_roc_auc"]:.4f} |
    | **Balanced Accuracy** | {train_metrics["train_balanced_accuracy"]:.1%} | {test_metrics["test_balanced_accuracy"]:.1%} | {cal_metrics["cal_balanced_accuracy"]:.1%} |
    | **F1 (macro)** | {train_metrics["train_f1_macro"]:.4f} | {test_metrics["test_f1_macro"]:.4f} | {cal_metrics["cal_f1_macro"]:.4f} |
    | **Precision (default)** | {train_metrics["train_precision_default"]:.4f} | {test_metrics["test_precision_default"]:.4f} | {cal_metrics["cal_precision_default"]:.4f} |
    | **Recall (default)** | {train_metrics["train_recall_default"]:.4f} | {test_metrics["test_recall_default"]:.4f} | {cal_metrics["cal_recall_default"]:.4f} |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
