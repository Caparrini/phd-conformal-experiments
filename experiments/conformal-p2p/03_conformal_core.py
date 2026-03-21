import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    EXPERIMENT_DIR = Path(__file__).parent
    CONFIG_DIR = str(EXPERIMENT_DIR / "config")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
    return EXPERIMENT_DIR, OmegaConf, cfg, mo


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

    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
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
        mlflow.log_figure(fig, "singleton_confusion_matrix.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig)
        import matplotlib.pyplot as plt
        from conformalpy.plots import (
            plot_singleton_confusion_matrix,
            plot_prediction_set_size_distribution,
            plot_set_size_vs_true_label,
        )

        # 1. Singleton confusion matrix
        fig, _ = plot_singleton_confusion_matrix(
            prediction_sets, y_test,
            class_names={0: "approve", 1: "reject"},
        )
        mlflow.log_figure(fig, "singleton_confusion_matrix.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig)

        # 2. Set size distribution
        fig, _ = plot_prediction_set_size_distribution(prediction_sets)
        type(fig)
        mlflow.log_figure(fig, "prediction_set_size_distribution.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig)

        # 3. Nonconformity scores
        # y_proba_test = pipeline.predict_proba(X_test)
        # scores_test = lac_nonconformity(y_test.values, y_proba_test)
        # fig, _ = plot_nonconformity_scores(
        #     y_cal, conf_clf._calibration_scores,
        #     y_test, scores_test,
        #     conf_clf.qhat_,
        # )
        # mlflow.log_figure(fig, "nonconformity_scores.png", save_kwargs={"bbox_inches": "tight"})
        # plt.close(fig)

        # 4. Set size vs true label
        fig, _ = plot_set_size_vs_true_label(prediction_sets, y_test)
        mlflow.log_figure(fig, "set_size_vs_true_label.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig)

        import numpy as np
        from conformalpy.plots import plot_coverage_by_alpha

        alphas = np.arange(0.01, 0.51, 0.01).tolist()

        # Una sola llamada — devuelve (n_samples, n_classes, n_alphas) bool array
        result = conf_clf.predict(X_test, alpha=alphas)

        coverages = []
        avg_sizes = []

        for a_idx in range(len(alphas)):
            sets_a = result[:, :, a_idx]  # (n_samples, n_classes) bool
            # Coverage: true label included?
            covered = sets_a[np.arange(len(y_test)), y_test.values.astype(int)]
            coverages.append(float(np.mean(covered)))
            # Avg set size
            avg_sizes.append(float(np.mean(sets_a.sum(axis=1))))

        fig_cov_by_alpha, _ = plot_coverage_by_alpha(alphas, coverages, widths=avg_sizes)
        mlflow.log_figure(fig_cov_by_alpha, "coverage_by_alpha.png", save_kwargs={"bbox_inches": "tight"})
        plt.close(fig_cov_by_alpha)

        import tempfile
        import os
        import json
    
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "prediction_sets.json")
            with open(filepath, "w") as f:
                json.dump(prediction_sets, f)
            mlflow.log_artifact(filepath, artifact_path="outputs")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
