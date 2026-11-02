"""Carga de configuración, particiones y modelos para los notebooks de un experimento.

Los notebooks de un mismo experimento repiten los mismos pasos: leer el yaml
con Hydra, partir los datos con la misma semilla, recuperar un modelo entrenado
de MLflow y calcular las mismas métricas. Aquí está cada paso una sola vez.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    log_loss,
    recall_score,
)

from dslib.data_utils import stratified_split


def load_cfg(experiment_dir: str | Path, name: str):
    """Lee ``<experiment_dir>/config/<name>.yaml`` con Hydra."""
    # Hydra exige una ruta absoluta.
    config_dir = str(Path(experiment_dir).resolve() / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=name)


def load_splits(cfg, experiment_dir: str | Path) -> dict:
    """Partición estratificada entrenamiento/calibración/test del experimento.

    Devuelve lo mismo que ``dslib.data_utils.stratified_split``. Con la misma
    semilla en ``cfg`` todos los notebooks ven exactamente los mismos préstamos.
    """
    features = list(cfg.data.features.numerical) + list(cfg.data.features.categorical)
    return stratified_split(
        data_path=Path(experiment_dir) / cfg.data.file,
        target=cfg.data.target,
        features=features,
        train_size=cfg.data.split.train_size,
        calibration_size=cfg.data.split.calibration_size,
        seed=cfg.experiment.seed,
    )


def make_preprocessor(cfg):
    """Numéricas tal cual y categóricas en *one-hot*, según las columnas de ``cfg``."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(
        [
            ("num", "passthrough", list(cfg.data.features.numerical)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(cfg.data.features.categorical),
            ),
        ]
    )


def load_run_model(cfg, run_name: str, trained_in: str):
    """Modelo del último run de MLflow con ese nombre.

    ``trained_in`` es el notebook que lo entrena; sale en el error si falta.
    """
    import mlflow

    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
    runs = (
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if experiment is not None
        else None
    )
    if runs is None or runs.empty:
        raise FileNotFoundError(
            f"No hay ningún run '{run_name}'. Ejecuta antes {trained_in}."
        )
    return mlflow.sklearn.load_model(f"runs:/{runs.iloc[0].run_id}/model")


def split_metrics(prefix: str, y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Log-loss, balanced accuracy y recall por clase, con ``prefix`` en cada clave."""
    n_classes = proba.shape[1]
    labels = list(range(n_classes))
    pred = proba.argmax(axis=1)
    recalls = recall_score(y, pred, average=None, labels=labels)
    out = {
        f"{prefix}log_loss": log_loss(y, proba, labels=labels),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y, pred),
    }
    out.update({f"{prefix}recall_{k}": float(r) for k, r in enumerate(recalls)})
    return out


def plot_confusion(y_true, y_pred, names: list[str], title: str):
    """Matriz de confusión normalizada por fila, con etiquetas en español.

    Cada fila suma 100 %: la diagonal es el recall de esa clase.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=names,
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    fig.tight_layout()
    return fig
