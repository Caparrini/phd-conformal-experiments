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
    return EXPERIMENT_DIR, cfg, mo, pl


@app.cell
def _(EXPERIMENT_DIR, cfg, pl):
    from datetime import date

    data_path = EXPERIMENT_DIR / cfg.data.file
    df = pl.read_parquet(data_path)

    train_cutoff = date.fromisoformat(cfg.data.splits.train_cutoff)
    cal_cutoff = date.fromisoformat(cfg.data.splits.calibration_cutoff)
    date_col = cfg.data.splits.date_column

    train_data = df.filter(pl.col(date_col) <= train_cutoff)
    cal_data = df.filter((pl.col(date_col) > train_cutoff) & (pl.col(date_col) <= cal_cutoff))
    test_data = df.filter(pl.col(date_col) > cal_cutoff)

    feature_cols = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)

    X_train = train_data.select(feature_cols).to_pandas()
    y_train = train_data[cfg.data.target].to_pandas()
    X_cal = cal_data.select(feature_cols).to_pandas()
    y_cal = cal_data[cfg.data.target].to_pandas()
    X_test = test_data.select(feature_cols).to_pandas()
    y_test = test_data[cfg.data.target].to_pandas()

    X_train.shape, X_cal.shape, X_test.shape
    return X_cal, X_test, y_cal, y_test


@app.cell
def _(cfg, mo):
    import dslib
    import mlflow

    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        max_results=1
    )
    model_id = models.iloc[0].model_id

    pipeline = mlflow.sklearn.load_model(f"models:/{model_id}")
    mo.md(f"Loaded model {model_id} from experiment {cfg.experiment.name}")
    return (pipeline,)


@app.cell
def _(X_cal, X_test, cfg, pipeline, y_cal, y_test):
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity.classification import lac_nonconformity
    from conformalpy.outcomes import categorize_outcomes, outcome_summary
    import numpy as np

    alpha = cfg.conformal.alpha

    conf_clf = ConformalClassifier(
        model=pipeline,
        alpha=alpha,
        nonconformity_function=lac_nonconformity,
        mondrian=True
    )

    conf_clf.calibrate(X_cal, y_cal)
    prediction_sets = conf_clf.predict(X_test)


    outcomes = categorize_outcomes(prediction_sets, y_test)
    outcome_stats = outcome_summary(prediction_sets, y_test)
    proportions = outcome_stats["proportions"]

    accuracy_confident = (
        f"{outcome_stats['accuracy_when_confident']:.1%}"
        if not np.isnan(outcome_stats["accuracy_when_confident"])
        else "N/A"
    )
    return (
        accuracy_confident,
        alpha,
        outcome_stats,
        prediction_sets,
        proportions,
    )


@app.cell
def _(
    accuracy_confident,
    alpha,
    mo,
    outcome_stats,
    prediction_sets,
    proportions,
    y_test,
):
    from conformalpy.evaluation import (
        classification_coverage_report,
        coverage_score,
        average_set_size,
    )
    import pandas as pd

    coverage = coverage_score(prediction_sets, y_test)
    avg_size = average_set_size(prediction_sets)
    coverage_report = classification_coverage_report(prediction_sets, y_test)

    mo.md(f"""
    ## Conformal Prediction Results (α={alpha})

    ### Outcome Distribution

    | Outcome | Proportion |
    |---|---|
    | **SC** (Singleton Correct) | {proportions['SC']:.1%} |
    | **SI** (Singleton Incorrect) | {proportions['SI']:.1%} |
    | **Multi-sets** (TS0+TS1) | {outcome_stats['uncertainty_rate']:.1%} |
    | **Empty** | {proportions['Empty']:.1%} |

    ### Coverage & Efficiency

    | Metric | Value |
    |---|---|
    | **Target coverage** | {1 - alpha:.1%} |
    | **Empirical coverage** | {coverage:.1%} |
    | **Coverage gap** | {coverage - (1 - alpha):+.2%} |
    | **Average set size** | {avg_size:.2f} |
    | **Confidence rate** (singletons) | {outcome_stats['confidence_rate']:.1%} |
    | **Accuracy when confident** | {accuracy_confident} |
    """)
    return


@app.cell
def _(prediction_sets, y_test):
    from conformalpy.outcomes import singleton_confusion_matrix
    from conformalpy.plots import plot_singleton_confusion_matrix
    import matplotlib.pyplot as plt


    scm = singleton_confusion_matrix(prediction_sets, y_test, class_names={0: "No Default", 1: "Default"})
    fig, axes = plot_singleton_confusion_matrix(
        prediction_sets, y_test,
        class_names={0: "approve", 1: "reject"}
    )
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
