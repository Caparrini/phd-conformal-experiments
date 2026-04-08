"""Tests for dslib.mlflow_config module.

Tests for configuration loading, validation, and fallback logic.
"""

import os
from unittest import mock

import pytest

from dslib.mlflow_config import (
    MLflowConfig,
    MLflowConnectivityError,
    _validate_tracking_uri,
    load_mlflow_config,
    reset_active_config,
)


class TestValidateTrackingUri:
    """Tests for _validate_tracking_uri()."""

    def test_local_file_uri_always_valid(self):
        """Local file:// URIs are always considered valid."""
        assert _validate_tracking_uri("file:.") is True
        assert _validate_tracking_uri("file:./mlruns") is True

    @mock.patch("requests.head")
    def test_remote_uri_accessible(self, mock_head):
        """Remote URI that responds successfully."""
        mock_head.return_value.status_code = 200
        assert _validate_tracking_uri("http://localhost:5000") is True
        mock_head.assert_called_once()

    @mock.patch("requests.head")
    def test_remote_uri_server_error(self, mock_head):
        """Remote URI that returns 5xx error."""
        mock_head.return_value.status_code = 500
        assert _validate_tracking_uri("http://localhost:5000") is False

    @mock.patch("requests.head")
    def test_remote_uri_connection_refused(self, mock_head):
        """Remote URI that connection is refused."""
        mock_head.side_effect = ConnectionError("Connection refused")
        assert _validate_tracking_uri("http://localhost:5000") is False

    @mock.patch("requests.head")
    def test_remote_uri_timeout(self, mock_head):
        """Remote URI that times out."""
        import requests
        mock_head.side_effect = requests.Timeout("Connection timeout")
        assert _validate_tracking_uri("http://localhost:5000") is False




class TestLoadMlflowConfig:
    """Tests for load_mlflow_config()."""

    @mock.patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_MODE": "flexible",
    }, clear=True)
    def test_load_remote_tracking_without_s3_endpoint(self):
        """Remote tracking without explicit S3 endpoint uses server-managed artifacts."""
        reset_active_config()
        config = load_mlflow_config()

        assert config.tracking_uri == "http://localhost:5000"
        assert config.mode == "flexible"
        assert config.artifact_store == "mlflow-server"

    @mock.patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://remote:5000",
        "MLFLOW_MODE": "remote",
    }, clear=True)
    def test_load_remote_mode(self):
        """Load config with remote mode uses mlflow-server for artifacts."""
        reset_active_config()
        config = load_mlflow_config()

        assert config.tracking_uri == "http://remote:5000"
        assert config.mode == "remote"
        assert config.artifact_store == "mlflow-server"

    @mock.patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_MODE": "invalid_mode",
    }, clear=True)
    def test_invalid_mode(self):
        """Invalid MLFLOW_MODE raises error."""
        reset_active_config()
        from dslib.mlflow_config import MLflowConfigError
        with pytest.raises(MLflowConfigError, match="Invalid MLFLOW_MODE"):
            load_mlflow_config()

    @mock.patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_MODE": "flexible",
    }, clear=True)
    def test_remote_tracking_uses_server_artifacts(self):
        """Remote tracking always uses mlflow-server for artifacts."""
        reset_active_config()
        config = load_mlflow_config()

        assert config.artifact_store == "mlflow-server"

    @mock.patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "file:./mlruns",
        "MLFLOW_MODE": "local",
    }, clear=True)
    def test_load_file_tracking_uses_local_artifact_store(self):
        """File-based tracking uses local artifact store."""
        reset_active_config()
        config = load_mlflow_config()

        assert config.artifact_store == "local"


class TestMlflowConfigValidation:
    """Tests for MLflowConfig.validate_and_log()."""

    @mock.patch("dslib.mlflow_config._validate_tracking_uri")
    @mock.patch("mlflow.set_tracking_uri")
    def test_validation_passes_local_tracking(self, mock_set_uri, mock_validate_uri):
        """Validation passes with local file-based tracking."""
        mock_validate_uri.return_value = True

        config = MLflowConfig(
            tracking_uri="file:./mlruns",
            mode="local",
            artifact_store="local",
        )
        # Should not raise
        config.validate_and_log()
        mock_set_uri.assert_called_once_with("file:./mlruns")

    @mock.patch("dslib.mlflow_config._validate_tracking_uri")
    def test_validation_fails_remote_mode_required(self, mock_validate_uri):
        """Validation fails when remote mode required but server unavailable."""
        mock_validate_uri.return_value = False

        config = MLflowConfig(
            tracking_uri="http://badremote:5000",
            mode="remote",
            artifact_store="local",
        )

        with pytest.raises(MLflowConnectivityError, match="Remote MLflow server"):
            config.validate_and_log()

    @mock.patch("dslib.mlflow_config._validate_tracking_uri")
    @mock.patch("mlflow.set_tracking_uri")
    def test_validation_fallback_flexible_mode(self, mock_set_uri, mock_validate_uri):
        """Validation allows fallback in flexible mode."""
        mock_validate_uri.return_value = False

        config = MLflowConfig(
            tracking_uri="http://badremote:5000",
            mode="flexible",
            artifact_store="local",
        )
        # Should not raise, just fallback
        config.validate_and_log()
        assert config.tracking_uri == "file:."
        assert config.artifact_store == "local"
        mock_set_uri.assert_called_once_with("file:.")

    @mock.patch("dslib.mlflow_config._validate_tracking_uri")
    def test_validation_passes_mlflow_server_artifacts(self, mock_validate_uri):
        """Validation passes for mlflow-server managed artifacts."""
        mock_validate_uri.return_value = True

        config = MLflowConfig(
            tracking_uri="http://minio:5000",
            mode="flexible",
            artifact_store="mlflow-server",
        )

        # Should not raise - artifacts managed by MLflow server
        config.validate_and_log()


class TestMlflowConfigGetOrCreateExperiment:
    """Tests for MLflowConfig.get_or_create_experiment()."""

    @mock.patch("dslib.mlflow_config.MlflowClient")
    @mock.patch("mlflow.set_tracking_uri")
    def test_experiment_exists(self, mock_set_uri, mock_client_class):
        """Get existing experiment by name."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        mock_experiment = mock.MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            mode="local",
            artifact_store="local",
        )
        exp_id = config.get_or_create_experiment("test-experiment")

        assert exp_id == "exp_123"
        mock_client.get_experiment_by_name.assert_called_once_with("test-experiment")
        mock_client.create_experiment.assert_not_called()

    @mock.patch("dslib.mlflow_config.MlflowClient")
    @mock.patch("mlflow.set_tracking_uri")
    def test_experiment_creates_new(self, mock_set_uri, mock_client_class):
        """Create new experiment if doesn't exist."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_456"

        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            mode="local",
            artifact_store="local",
        )
        exp_id = config.get_or_create_experiment("new-experiment")

        assert exp_id == "exp_456"
        mock_client.get_experiment_by_name.assert_called_once_with("new-experiment")
        mock_client.create_experiment.assert_called_once_with("new-experiment")


@mock.patch.dict(os.environ, {
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "MLFLOW_MODE": "flexible",
})
def test_global_active_config_singleton():
    """Test that global active config is a singleton."""
    from dslib.mlflow_config import get_active_config

    reset_active_config()

    config1 = get_active_config()
    config2 = get_active_config()

    assert config1 is config2
