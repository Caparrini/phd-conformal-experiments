"""Centralized MLflow tracking configuration with validation.

Provides unified MLflow setup from environment variables with flexible fallback
and upfront validation to prevent silent failures. Artifact storage is managed
by MLflow server; this module only configures the tracking connection.

Quick Start:
    config = load_mlflow_config()
    config.validate_and_log()  # Validates server connectivity upfront
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import requests
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import Run


# Global singleton instance of MLflowConfig (lazy-loaded)
_ACTIVE_CONFIG: Optional["MLflowConfig"] = None


class MLflowConfigError(Exception):
    """Raised when MLflow configuration is invalid or unavailable."""
    pass


class MLflowConnectivityError(MLflowConfigError):
    """Raised when unable to connect to MLflow tracking server."""
    pass


@dataclass
class MLflowConfig:
    """MLflow tracking configuration.
    
    Attributes:
        tracking_uri: MLflow server URI (http://host:port or file:/path).
        mode: Fallback strategy:
            - "local": only file-based tracking (file://...)
            - "remote": require server connection, fail if unavailable
            - "flexible": try server first, fallback to local if needed
        artifact_store: Inferred artifact location:
            - "local": file-based tracking (file:// URI)
            - "mlflow-server": remote server manages artifacts
    """
    
    tracking_uri: str
    mode: Literal["local", "remote", "flexible"]
    artifact_store: Literal["local", "mlflow-server"]
    _client: Optional[MlflowClient] = None
    
    def get_client(self) -> MlflowClient:
        """Get or create MLflow client."""
        if self._client is None:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client
    
    def validate_and_log(self) -> None:
        """Validate tracking server connectivity and log active config.
        
        Raises:
            MLflowConnectivityError: If remote server unreachable and mode requires it.
        """
        logger.info("[MLflow] Validating configuration...")
        
        # Validate tracking URI (critical - without it, MLflow cannot function)
        if not _validate_tracking_uri(self.tracking_uri):
            if self.mode == "remote":
                msg = (
                    f"[MLflow] ERROR: Remote MLflow server at {self.tracking_uri} "
                    "is not accessible and mode='remote' requires it.\n"
                    "Check: 1) Server is running  2) URI is correct  3) Network connectivity"
                )
                logger.error(msg)
                raise MLflowConnectivityError(msg)
            elif self.mode == "flexible":
                logger.warning(
                    f"[MLflow] Remote MLflow server at {self.tracking_uri} "
                    "is not accessible. Falling back to local tracking."
                )
                self.tracking_uri = "file:."
                self.artifact_store = "local"

        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Log final configuration
        self._log_active_configuration()
    
    def _log_active_configuration(self) -> None:
        """Log the active configuration for debugging."""
        logger.info("[MLflow] ========== Active MLflow Configuration ==========")
        logger.info(f"[MLflow] Tracking URI: {self.tracking_uri}")
        logger.info(f"[MLflow] Mode: {self.mode}")
        logger.info(f"[MLflow] Artifact Store: {self.artifact_store}")
        logger.info("[MLflow] ====================================================")
    
    def get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create an MLflow experiment by name.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Experiment ID (string)
        """
        # Apply tracking URI before creating client
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        client = self.get_client()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            logger.debug(f"[MLflow] Created experiment '{experiment_name}' (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.debug(f"[MLflow] Using existing experiment '{experiment_name}' (ID: {experiment_id})")
        
        return experiment_id


def load_mlflow_config() -> MLflowConfig:
    """Load MLflow configuration from environment variables.
    
    Environment Variables (from .env):
        MLFLOW_TRACKING_URI: Server URI or file path (default: http://localhost:5000).
        MLFLOW_MODE: Fallback mode "local", "remote", or "flexible" (default: flexible).
    
    Returns:
        MLflowConfig instance with tracking URI and mode.
        
    Raises:
        MLflowConfigError: If MLFLOW_MODE is invalid.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mode = os.getenv("MLFLOW_MODE", "flexible")
    
    if mode not in ("local", "remote", "flexible"):
        raise MLflowConfigError(
            f"Invalid MLFLOW_MODE '{mode}'. Must be 'local', 'remote', or 'flexible'."
        )
    
    # Infer artifact store from tracking URI
    artifact_store = "local" if tracking_uri.startswith("file:") else "mlflow-server"
    
    return MLflowConfig(
        tracking_uri=tracking_uri,
        mode=mode,
        artifact_store=artifact_store,
    )


def get_active_config() -> MLflowConfig:
    """Get the global active MLflow configuration instance.
    
    Lazily loads configuration on first call.
    
    Returns:
        Global MLflowConfig instance
    """
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = load_mlflow_config()
    return _ACTIVE_CONFIG


def reset_active_config() -> None:
    """Reset the global active configuration (mainly for testing)."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None


def _validate_tracking_uri(tracking_uri: str) -> bool:
    """Check if MLflow tracking server is accessible.
    
    Args:
        tracking_uri: Tracking server URI
        
    Returns:
        True if server is accessible, False otherwise
    """
    if tracking_uri.startswith("file:"):
        # Local file-based tracking is always "available"
        return True
    
    try:
        # Try to reach the MLflow server with a timeout
        response = requests.head(
            tracking_uri,
            timeout=5,
            allow_redirects=True,
        )
        return response.status_code < 500
    except (requests.RequestException, Exception):
        return False




# ============================================================================
# HELPER FUNCTIONS - Reutilizable logic for experiment scripts
# ============================================================================


def search_runs_by_experiment_and_name(
    experiment_name: str,
    run_name: str,
    mlflow_config: Optional[MLflowConfig] = None,
) -> list[Run]:
    """Search for MLflow runs by experiment name and run name.
    
    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name (exact match)
        mlflow_config: Optional config instance. If None, uses global config.
        
    Returns:
        List of runs matching the criteria
        
    Raises:
        MLflowConfigError: If experiment not found
    """
    if mlflow_config is None:
        mlflow_config = get_active_config()
    
    client = mlflow_config.get_client()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise MLflowConfigError(
            f"Experiment '{experiment_name}' not found. "
            "Check if it was created by a previous step."
        )
    
    # Search runs with filter
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    
    return runs


def load_model_from_registry(
    model_uri: str,
    flavor: str = "sklearn",
    mlflow_config: Optional[MLflowConfig] = None,
) -> Any:
    """Load a model from MLflow model registry.
    
    Args:
        model_uri: Model URI (e.g., "models:/model_name/production")
        flavor: Model flavor (sklearn, xgboost, etc.)
        mlflow_config: Optional config instance
        
    Returns:
        Loaded model
        
    Raises:
        MLflowConfigError: If model not found or loading fails
    """
    if mlflow_config is None:
        mlflow_config = get_active_config()
    
    import mlflow
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        if flavor == "sklearn":
            return mlflow.sklearn.load_model(model_uri)
        elif flavor == "xgboost":
            return mlflow.xgboost.load_model(model_uri)
        else:
            return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise MLflowConfigError(
            f"Failed to load model '{model_uri}': {e}"
        )


def download_artifact_safe(
    run_id: str,
    artifact_path: str,
    mlflow_config: Optional[MLflowConfig] = None,
) -> str:
    """Download an artifact from a run, with error handling.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to artifact within run
        mlflow_config: Optional config instance
        
    Returns:
        Local path to downloaded artifact
        
    Raises:
        MLflowConfigError: If download fails
    """
    if mlflow_config is None:
        mlflow_config = get_active_config()
    
    import mlflow
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        return local_path
    except Exception as e:
        raise MLflowConfigError(
            f"Failed to download artifact '{artifact_path}' from run '{run_id}': {e}"
        )
