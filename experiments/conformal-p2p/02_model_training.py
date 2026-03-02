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
    return data_path, df, pl


@app.cell
def _(cfg, df, mo, pl):
    from datetime import date

    train_cutoff = date.fromisoformat(cfg.data.splits.train_cutoff)
    cal_cutoff = date.fromisoformat(cfg.data.splits.calibration_cutoff)
    date_col = cfg.data.splits.date_column

    train_data = df.filter(pl.col(date_col) <= train_cutoff)
    cal_data = df.filter((pl.col(date_col) > train_cutoff) & (pl.col(date_col) <= cal_cutoff))
    test_data = df.filter(pl.col(date_col) > cal_cutoff)

    total = df.shape[0]

    mo.md(f"""
    ## Splits

    | Split | Rows | Default Rate | Date From | Date To | % of Total |
    |---|---|---|---|---|---|
    | Train | {train_data.shape[0]:,} | {train_data[cfg.data.target].mean():.1%} | {str(train_data[date_col].min())} | {str(train_data[date_col].max())} | {train_data.shape[0]/total:.1%} |
    | Calibration | {cal_data.shape[0]:,} | {cal_data[cfg.data.target].mean():.1%} | {str(cal_data[date_col].min())} | {str(cal_data[date_col].max())} | {cal_data.shape[0]/total:.1%} |
    | Test | {test_data.shape[0]:,} | {test_data[cfg.data.target].mean():.1%} | {str(test_data[date_col].min())} | {str(test_data[date_col].max())} | {test_data.shape[0]/total:.1%} |
    """)
    return cal_data, test_data, train_data


@app.cell
def _(cal_data, cfg, test_data, train_data):
    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    X_train = train_data.select(feature_cols).to_pandas()
    y_train = train_data[cfg.data.target].to_pandas()

    X_cal = cal_data.select(feature_cols).to_pandas()
    y_cal = cal_data[cfg.data.target].to_pandas()

    X_test = test_data.select(feature_cols).to_pandas()
    y_test = test_data[cfg.data.target].to_pandas()

    X_train.shape, X_cal.shape, X_test.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(cfg):
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    numerical_cols = list(cfg.data.features.numerical) 
    categorical_cols = list(cfg.data.features.categorical)

    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", StandardScaler(), numerical_cols), # It is not necessary for XGBoost
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    model = XGBClassifier(
        **dict(cfg.model.params),
        n_jobs=-1,
        verbosity=0
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return (pipeline,)


@app.cell
def _(
    OmegaConf,
    X_test,
    X_train,
    cfg,
    data_path,
    mo,
    pipeline,
    y_test,
    y_train,
):
    import mlflow
    from dslib.tracking import tracked_run

    mlflow.autolog()

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tracked_run(config_dict, data_path, cfg.experiment.name, run_name="xgboost-baseline"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("test_balanced_accuracy", balanced_accuracy_score(y_test, y_pred))

    mo.md(f"""
    ## Training Complete

    - **Test ROC-AUC**: {roc_auc_score(y_test, y_proba):.4f}
    - **Test Balanced Accuracy**: {balanced_accuracy_score(y_test, y_pred):.1%}
    - **MLflow run**: {mlflow.active_run() is not None}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
