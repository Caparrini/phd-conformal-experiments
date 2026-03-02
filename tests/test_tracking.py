"""Tests for dslib.tracking."""

import hashlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from dslib.tracking import (
    _compute_file_hash,
    _get_git_commit,
    _has_uncommitted_changes,
    _log_params_flat,
    _to_dict,
    tracked_run,
)


# ---------------------------------------------------------------------------
# _to_dict
# ---------------------------------------------------------------------------
class TestToDict:
    def test_plain_dict(self):
        d = {"a": 1, "b": 2}
        assert _to_dict(d) == {"a": 1, "b": 2}

    def test_dataclass(self):
        @dataclass
        class Cfg:
            lr: float = 0.01
            epochs: int = 10

        result = _to_dict(Cfg())
        assert result == {"lr": 0.01, "epochs": 10}

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported config type"):
            _to_dict("not a config")

    def test_omegaconf_dictconfig(self):
        """Only runs if omegaconf is installed."""
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.create({"model": {"name": "xgb", "depth": 5}})
            result = _to_dict(cfg)
            assert result == {"model": {"name": "xgb", "depth": 5}}
        except ImportError:
            pytest.skip("omegaconf not installed")


# ---------------------------------------------------------------------------
# _log_params_flat
# ---------------------------------------------------------------------------
class TestLogParamsFlat:
    @patch("dslib.tracking.mlflow")
    def test_simple_params(self, mock_mlflow):
        _log_params_flat({"lr": 0.01, "epochs": 10})

        mock_mlflow.log_param.assert_any_call("lr", 0.01)
        mock_mlflow.log_param.assert_any_call("epochs", 10)
        assert mock_mlflow.log_param.call_count == 2

    @patch("dslib.tracking.mlflow")
    def test_nested_params(self, mock_mlflow):
        _log_params_flat({"model": {"name": "xgb", "depth": 5}})

        mock_mlflow.log_param.assert_any_call("model.name", "xgb")
        mock_mlflow.log_param.assert_any_call("model.depth", 5)
        assert mock_mlflow.log_param.call_count == 2

    @patch("dslib.tracking.mlflow")
    def test_list_params(self, mock_mlflow):
        _log_params_flat({"alpha": [0.05, 0.10, 0.20]})

        mock_mlflow.log_param.assert_called_once_with(
            "alpha", "[0.05, 0.1, 0.2]"
        )

    @patch("dslib.tracking.mlflow")
    def test_empty_config(self, mock_mlflow):
        _log_params_flat({})

        mock_mlflow.log_param.assert_not_called()


# ---------------------------------------------------------------------------
# _get_git_commit
# ---------------------------------------------------------------------------
class TestGetGitCommit:
    @patch("dslib.tracking.subprocess.run")
    def test_returns_commit_hash(self, mock_run):
        mock_run.return_value = MagicMock(stdout="abc123def456\n")
        assert _get_git_commit() == "abc123def456"

    @patch(
        "dslib.tracking.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    )
    def test_returns_unknown_when_git_missing(self, mock_run):
        assert _get_git_commit() == "unknown"


# ---------------------------------------------------------------------------
# _has_uncommitted_changes
# ---------------------------------------------------------------------------
class TestHasUncommittedChanges:
    @patch("dslib.tracking.subprocess.run")
    def test_clean_repo(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        assert _has_uncommitted_changes() is False

    @patch("dslib.tracking.subprocess.run")
    def test_dirty_repo(self, mock_run):
        mock_run.return_value = MagicMock(stdout=" M src/train.py\n")
        assert _has_uncommitted_changes() is True

    @patch(
        "dslib.tracking.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    )
    def test_returns_false_when_git_missing(self, mock_run):
        assert _has_uncommitted_changes() is False


# ---------------------------------------------------------------------------
# _compute_file_hash
# ---------------------------------------------------------------------------
class TestComputeFileHash:
    def test_correct_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _compute_file_hash(f) == expected


# ---------------------------------------------------------------------------
# tracked_run (integration)
# ---------------------------------------------------------------------------
class TestTrackedRun:
    @patch("dslib.tracking._has_uncommitted_changes", return_value=False)
    @patch("dslib.tracking._get_git_commit", return_value="abc123")
    @patch("dslib.tracking.mlflow")
    def test_logs_all_metadata(self, mock_mlflow, mock_git, mock_dirty, tmp_path):
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"fake parquet data")

        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        config = {"model": {"name": "xgb"}, "alpha": 0.10}

        with tracked_run(config, data_file, "test-experiment", "test-run"):
            pass

        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

        tag_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.set_tag.call_args_list
        }
        assert tag_calls["git_commit"] == "abc123"
        assert tag_calls["git_dirty"] is False
        assert tag_calls["data_path"] == str(data_file)
        assert "data_hash" in tag_calls

        param_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.log_param.call_args_list
        }
        assert param_calls["model.name"] == "xgb"
        assert param_calls["alpha"] == 0.10

    @patch("dslib.tracking._has_uncommitted_changes", return_value=True)
    @patch("dslib.tracking._get_git_commit", return_value="abc123")
    @patch("dslib.tracking.mlflow")
    def test_warns_on_uncommitted_changes(
        self, mock_mlflow, mock_git, mock_dirty, tmp_path, caplog
    ):
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"fake data")

        mock_run = MagicMock()
        mock_run.info.run_id = "run-456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with tracked_run({"seed": 42}, data_file, "exp"):
            pass

        tag_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.set_tag.call_args_list
        }
        assert tag_calls["git_dirty"] is True

    @patch("dslib.tracking._has_uncommitted_changes", return_value=False)
    @patch("dslib.tracking._get_git_commit", return_value="abc123")
    @patch("dslib.tracking.mlflow")
    def test_handles_missing_data_file(
        self, mock_mlflow, mock_git, mock_dirty, tmp_path
    ):
        missing_file = tmp_path / "nonexistent.parquet"

        mock_run = MagicMock()
        mock_run.info.run_id = "run-789"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with tracked_run({"seed": 42}, missing_file, "exp"):
            pass

        tag_calls = {
            call.args[0]: call.args[1]
            for call in mock_mlflow.set_tag.call_args_list
        }
        assert tag_calls["data_hash"] == "file_not_found"