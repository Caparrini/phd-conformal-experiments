"""MLflow experiment tracking wrapper.

Complements mlflow.autolog() with metadata that MLflow doesn't capture
automatically when NOT using MLflow Projects.

What mlflow.sklearn.autolog() handles (verified for MLflow >=2.x, <=3.9):
- Model parameters via estimator.get_params(deep=True)
- Training metrics: precision, recall, f1, accuracy, roc_auc, log_loss
- Model artifact serialization (cloudpickle)
- Model signature and input examples (if enabled)
- Estimator class tags

What autolog does NOT handle (and this wrapper covers):
- Git commit hash (only logged automatically via MLflow Projects, not scripts/notebooks)
- Git dirty status (uncommitted changes warning)
- SHA-256 hash of the input data file (exact data provenance)
- Experiment config from Hydra (non-model parameters like alpha, split ratios)

References:
- autolog docs: https://mlflow.org/docs/latest/tracking/autolog.html
- sklearn autolog: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html
- git commit only in Projects: https://github.com/mlflow/mlflow/discussions/7995
- Hydra Compose API: https://hydra.cc/docs/advanced/compose_api/
"""

import hashlib
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not get git commit hash")
        return "unknown"


def _has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes in the repo."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_dict(config: Any) -> dict:
    """Convert a config object to a plain dict.

    Supports Hydra/OmegaConf DictConfig, dataclasses, and plain dicts.
    """
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass

    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(config)

    if isinstance(config, dict):
        return config

    raise TypeError(f"Unsupported config type: {type(config)}")


def _log_params_flat(config: dict, prefix: str = "") -> None:
    """Recursively flatten and log a config dict as MLflow params."""
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _log_params_flat(value, prefix=full_key)
        elif isinstance(value, (list, tuple)):
            mlflow.log_param(full_key, str(value))
        else:
            mlflow.log_param(full_key, value)


@contextmanager
def tracked_run(
    config: Any,
    data_path: str | Path,
    experiment_name: str,
    run_name: str | None = None,
):
    """Context manager that adds data provenance and config to an MLflow run.

    Use together with mlflow.autolog(), which handles model parameters,
    metrics, and artifacts automatically.

    This wrapper adds:
    - Git commit hash and dirty status
    - Data file hash for reproducibility
    - Hydra/experiment config as params

    Accepts Hydra DictConfig, dataclasses, or plain dicts as config.

    Args:
        config: Experiment configuration (Hydra DictConfig, dataclass, or dict).
        data_path: Path to the input data file.
        experiment_name: MLflow experiment name.
        run_name: Optional name for this specific run.

    Yields:
        The active mlflow.ActiveRun.

    Example using Hydra Compose API (e.g. in marimo):
        from hydra import initialize, compose

        with initialize(config_path="configs"):
            cfg = compose(config_name="config")

        mlflow.autolog()
        with tracked_run(cfg, "data/cleaned/lending_club.parquet", "conformal-p2p"):
            pipeline.fit(X_train, y_train)
    """
    data_path = Path(data_path)

    git_dirty = _has_uncommitted_changes()
    if git_dirty:
        logger.warning(
            "There are uncommitted changes. Results may not be fully reproducible."
        )

    git_commit = _get_git_commit()
    data_hash = (
        _compute_file_hash(data_path) if data_path.exists() else "file_not_found"
    )
    config_dict = _to_dict(config)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("git_dirty", git_dirty)
        mlflow.set_tag("data_path", str(data_path))
        mlflow.set_tag("data_hash", data_hash)

        _log_params_flat(config_dict)

        logger.info(
            f"Started MLflow run: {run.info.run_id} "
            f"(experiment: {experiment_name}, commit: {git_commit[:8]})"
        )

        yield run

    logger.info(f"Finished MLflow run: {run.info.run_id}")