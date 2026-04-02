"""Tests for dslib.shap_cache — joblib.Memory-based KernelSHAP caching."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dslib.shap_cache import compute_or_load_shap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
X_BG = RNG.standard_normal((10, 3)).astype(np.float64)
X_EX = RNG.standard_normal((5, 3)).astype(np.float64)
FAKE_SHAP = RNG.standard_normal((5, 3)).astype(np.float64)


def _make_mock_explainer(shap_output: np.ndarray) -> MagicMock:
    """Return a mock KernelExplainer whose shap_values returns shap_output."""
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = shap_output
    mock_constructor = MagicMock(return_value=mock_explainer)
    return mock_constructor


def _dummy_predict(X: np.ndarray) -> np.ndarray:
    return X @ np.ones((3, 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_computes_and_creates_cache(tmp_path: Path) -> None:
    """First call should compute SHAP and populate the cache directory."""
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        result = compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )

    np.testing.assert_array_equal(result, FAKE_SHAP)
    assert mock_constructor.call_count == 1
    # joblib creates subdirectories under cache_dir
    assert any(tmp_path.iterdir())


def test_cache_hit_skips_computation(tmp_path: Path) -> None:
    """Second call with same args must return cached result without recomputing."""
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        v1 = compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )
        v2 = compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )

    np.testing.assert_array_equal(v1, v2)
    assert mock_constructor.call_count == 1  # only one computation


def test_cache_miss_on_model_id_change(tmp_path: Path) -> None:
    """Changing model_id must trigger recomputation."""
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-b", cache_dir=tmp_path,
        )

    assert mock_constructor.call_count == 2


def test_cache_miss_on_nsamples_change(tmp_path: Path) -> None:
    """Changing nsamples must trigger recomputation."""
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=50, model_id="model-a", cache_dir=tmp_path,
        )

    assert mock_constructor.call_count == 2


def test_cache_miss_on_different_x_background_size(tmp_path: Path) -> None:
    """Different X_background (different rows) must trigger recomputation."""
    rng = np.random.default_rng(99)
    X_bg_other = rng.standard_normal((8, 3)).astype(np.float64)
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )
        compute_or_load_shap(
            _dummy_predict, X_bg_other, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )

    assert mock_constructor.call_count == 2


def test_force_bypasses_valid_cache(tmp_path: Path) -> None:
    """force=True must recompute even when a cached entry exists."""
    mock_constructor = _make_mock_explainer(FAKE_SHAP)

    with patch("shap.KernelExplainer", mock_constructor):
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
        )
        compute_or_load_shap(
            _dummy_predict, X_BG, X_EX,
            nsamples=10, model_id="model-a", cache_dir=tmp_path,
            force=True,
        )

    assert mock_constructor.call_count == 2


def test_cache_dir_created_if_missing() -> None:
    """cache_dir does not need to exist beforehand."""
    with tempfile.TemporaryDirectory() as td:
        nonexistent = Path(td) / "new" / "nested" / "cache"
        mock_constructor = _make_mock_explainer(FAKE_SHAP)

        with patch("shap.KernelExplainer", mock_constructor):
            result = compute_or_load_shap(
                _dummy_predict, X_BG, X_EX,
                nsamples=10, model_id="model-a", cache_dir=nonexistent,
            )

        assert nonexistent.exists()
        np.testing.assert_array_equal(result, FAKE_SHAP)
