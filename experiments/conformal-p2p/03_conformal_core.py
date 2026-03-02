import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
    return EXPERIMENT_DIR, OmegaConf, cfg, mo, pl


@app.cell
def _(EXPERIMENT_DIR, cfg, pl):
    from datetime import date

    data_path = EXPERIMENT_DIR / cfg.data.file
    df = pl.read_parquet(data_path)

    train_cutoff = date.fromisoformat(cfg.data.splits.train_cutoff)
    cal_cutoff = date.fromisoformat(cfg.data.splits.calibration_cutoff)
    date_col = cfg.data.splits.date_column

    train_data = df.filter(pl.col(date_col) <= train_cutoff)
    cal_data = df.filter(
        (pl.col(date_col) > train_cutoff) & (pl.col(date_col) <= cal_cutoff)
    )
    test_data = df.filter(pl.col(date_col) > cal_cutoff)

    feature_cols = list(cfg.data.features.numerical) + list(
        cfg.data.features.categorical
    )

    X_train = train_data.select(feature_cols).to_pandas()
    y_train = train_data[cfg.data.target].to_pandas()
    X_cal = cal_data.select(feature_cols).to_pandas()
    y_cal = cal_data[cfg.data.target].to_pandas()
    X_test = test_data.select(feature_cols).to_pandas()
    y_test = test_data[cfg.data.target].to_pandas()

    X_train.shape, X_cal.shape, X_test.shape
    return X_cal, X_test, data_path, y_cal, y_test


@app.cell
def _(cfg, mo):
    import mlflow

    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string="name = 'model'",
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        max_results=1,
    )
    model_id = models.iloc[0].model_id

    pipeline = mlflow.sklearn.load_model(f"models:/{model_id}")
    mo.md(f"Loaded model {model_id} from experiment {cfg.experiment.name}")
    return mlflow, pipeline


@app.cell
def _(
    OmegaConf,
    X_cal,
    X_test,
    cfg,
    data_path,
    mlflow,
    pipeline,
    y_cal,
    y_test,
):
    from dslib.tracking import tracked_run
    from conformalpy.integrations.mlflow import ConformalMLflowCallback
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.outcomes import (
        categorize_outcomes,
        outcome_summary,
    )
    from conformalpy.plots import plot_singleton_confusion_matrix
    from conformalpy.evaluation import coverage_score
    from conformalpy.nonconformity import lac_nonconformity
    import matplotlib.pyplot as plt

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    callback = ConformalMLflowCallback()

    alpha = cfg.conformal.alpha

    with tracked_run(
        config_dict, data_path, cfg.experiment.name, run_name="conformal-evaluation"
    ):
        conf_clf = ConformalClassifier(
            model=pipeline,
            alpha=alpha,
            nonconformity_function=lac_nonconformity,
            mondrian=True,
            callbacks=[callback],
        )
        conf_clf.calibrate(X_cal, y_cal)
        conf_clf.score(X_test, y_test)
        prediction_sets = conf_clf.predict(X_test)

        # Outcomes
        outcomes = categorize_outcomes(prediction_sets, y_test)
        outcome_stats = outcome_summary(prediction_sets, y_test)
        proportions = outcome_stats["proportions"]
        coverage = coverage_score(prediction_sets, y_test)

        # Plots as artifacts
        fig, axes = plot_singleton_confusion_matrix(
            prediction_sets,
            y_test,
            class_names={0: "approve", 1: "reject"},
        )
        mlflow.log_figure(fig, "singleton_confusion_matrix.png")
        plt.close(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
