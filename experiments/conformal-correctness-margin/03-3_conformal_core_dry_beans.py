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
        cfg = compose(config_name="dry_beans")

    return EXPERIMENT_DIR, OmegaConf, cfg, mo


@app.cell
def _(EXPERIMENT_DIR, cfg):
    from dslib.data_utils import stratified_split
    from sklearn.preprocessing import LabelEncoder

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

    X_train, y_train_raw = splits["train"]
    X_cal, y_cal_raw = splits["calibration"]
    X_test, y_test_raw = splits["test"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_cal = le.transform(y_cal_raw)
    y_test = le.transform(y_test_raw)

    class_names = {i: name for i, name in enumerate(le.classes_)}

    return X_cal, X_test, class_names, data_path, y_cal, y_test


@app.cell
def _(cfg, mo):
    import mlflow

    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{cfg.experiment.run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0].run_id
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    mo.md(f"Loaded model from run `{run_id}` ({cfg.experiment.run_name})")
    return mlflow, pipeline


@app.cell
def _(
    OmegaConf,
    X_cal,
    X_test,
    cfg,
    class_names,
    data_path,
    mlflow,
    pipeline,
    y_cal,
    y_test,
):
    import json
    import os
    import tempfile
    import numpy as np
    import matplotlib.pyplot as plt
    from dslib.tracking import tracked_run
    from conformalpy.classifier import ConformalClassifier
    from conformalpy.nonconformity import lac_nonconformity
    from conformalpy.plots import (
        plot_prediction_set_size_distribution,
        plot_set_size_vs_true_label,
        plot_coverage_by_alpha,
    )
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    import seaborn as sns
    from conformalpy.integrations.mlflow import ConformalMLflowCallback

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    alpha = cfg.conformal.alpha

    with tracked_run(
        config_dict, data_path, cfg.experiment.name,
        run_name="dry-beans-conformal-evaluation",
    ):
        conf_clf = ConformalClassifier(
            model=pipeline,
            alpha=alpha,
            nonconformity_function=lac_nonconformity,
            mondrian=True,
            callbacks=[ConformalMLflowCallback()],
        )
        conf_clf.calibrate(X_cal, y_cal)
        conf_clf.score(X_test, y_test)
        prediction_sets = conf_clf.predict(X_test)

        # 1. Singleton confusion matrix (N×N multiclass)
        singleton_mask = [len(s) == 1 for s in prediction_sets]
        y_true_sing = np.asarray(y_test)[singleton_mask]
        y_pred_sing = np.array([next(iter(s)) for s, m in zip(prediction_sets, singleton_mask) if m])
        n_sing = int(np.sum(singleton_mask))
        labels = list(class_names.keys())
        label_names = [class_names[k] for k in labels]
        cm = sk_confusion_matrix(y_true_sing, y_pred_sing, labels=labels)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted class (singleton)")
        ax.set_ylabel("True class")
        ax.set_title(f"Singleton Confusion Matrix — {n_sing}/{len(prediction_sets)} singletons")
        plt.tight_layout()
        mlflow.log_figure(fig, "singleton_confusion_matrix.png")
        plt.close(fig)

        # 2. Set size distribution
        fig, _ = plot_prediction_set_size_distribution(prediction_sets)
        mlflow.log_figure(fig, "prediction_set_size_distribution.png")
        plt.close(fig)

        # 3. Set size vs true label
        fig, _ = plot_set_size_vs_true_label(prediction_sets, y_test)
        mlflow.log_figure(fig, "set_size_vs_true_label.png")
        plt.close(fig)

        # 4. Coverage by alpha (range 0.01..0.50)
        alphas = np.arange(0.01, 0.51, 0.01).tolist()
        result = conf_clf.predict(X_test, alpha=alphas)  # (n_samples, n_classes, n_alphas)
        y_int = np.asarray(y_test).astype(int)
        coverages = [
            float(np.mean(result[:, :, i][np.arange(len(y_int)), y_int]))
            for i in range(len(alphas))
        ]
        avg_sizes = [
            float(np.mean(result[:, :, i].sum(axis=1)))
            for i in range(len(alphas))
        ]
        fig, _ = plot_coverage_by_alpha(alphas, coverages, widths=avg_sizes)
        mlflow.log_figure(fig, "coverage_by_alpha.png")
        plt.close(fig)

        # 5. Prediction sets artifact
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
