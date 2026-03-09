"""Tests for dslib.data_utils."""

import hashlib
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dslib.data_utils import download_if_missing, stratified_split


@pytest.fixture
def tmp_dest(tmp_path):
    """Return a temporary destination path for downloads."""
    return tmp_path / "data" / "test_file.csv"


def _fake_download(url, dest):
    """Simulate a download by writing dummy content."""
    Path(dest).write_text("fake,data\n1,2\n")


def _sha256(content: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(content.encode()).hexdigest()


FAKE_CONTENT = "fake,data\n1,2\n"
FAKE_HASH = _sha256(FAKE_CONTENT)


class TestDownloadIfMissing:
    """Tests for download_if_missing."""

    @patch("dslib.data_utils.urllib.request.urlretrieve", side_effect=_fake_download)
    def test_downloads_when_file_missing(self, mock_urlretrieve, tmp_dest):
        result = download_if_missing("https://example.com/data.csv", tmp_dest)

        assert result == tmp_dest
        assert tmp_dest.exists()
        mock_urlretrieve.assert_called_once()

    @patch("dslib.data_utils.urllib.request.urlretrieve")
    def test_skips_download_when_file_exists(self, mock_urlretrieve, tmp_dest):
        tmp_dest.parent.mkdir(parents=True)
        tmp_dest.write_text(FAKE_CONTENT)

        result = download_if_missing("https://example.com/data.csv", tmp_dest)

        assert result == tmp_dest
        mock_urlretrieve.assert_not_called()

    @patch("dslib.data_utils.urllib.request.urlretrieve")
    def test_skips_download_when_hash_matches(self, mock_urlretrieve, tmp_dest):
        tmp_dest.parent.mkdir(parents=True)
        tmp_dest.write_text(FAKE_CONTENT)

        result = download_if_missing(
            "https://example.com/data.csv", tmp_dest, sha256=FAKE_HASH
        )

        assert result == tmp_dest
        mock_urlretrieve.assert_not_called()

    @patch(
        "dslib.data_utils.urllib.request.urlretrieve",
        side_effect=_fake_download,
    )
    def test_redownloads_when_hash_mismatch(self, mock_urlretrieve, tmp_dest):
        tmp_dest.parent.mkdir(parents=True)
        tmp_dest.write_text("old,corrupted,content\n")

        result = download_if_missing(
            "https://example.com/data.csv", tmp_dest, sha256=FAKE_HASH
        )

        assert result == tmp_dest
        assert tmp_dest.read_text() == FAKE_CONTENT
        mock_urlretrieve.assert_called_once()

    @patch(
        "dslib.data_utils.urllib.request.urlretrieve",
        side_effect=HTTPError(
            "https://example.com/data.csv", 404, "Not Found", {}, None
        ),
    )
    def test_raises_on_http_error(self, mock_urlretrieve, tmp_dest):
        with pytest.raises(HTTPError):
            download_if_missing("https://example.com/data.csv", tmp_dest)

    @patch(
        "dslib.data_utils.urllib.request.urlretrieve",
        side_effect=URLError("Name resolution failed"),
    )
    def test_raises_on_url_error(self, mock_urlretrieve, tmp_dest):
        with pytest.raises(URLError):
            download_if_missing("https://example.com/data.csv", tmp_dest)

    @patch("dslib.data_utils.urllib.request.urlretrieve", side_effect=_fake_download)
    def test_raises_on_hash_mismatch_after_download(self, mock_urlretrieve, tmp_dest):
        with pytest.raises(ValueError, match="Hash mismatch"):
            download_if_missing(
                "https://example.com/data.csv", tmp_dest, sha256="wrong_hash"
            )

        assert not tmp_dest.exists(), "File should be deleted after hash mismatch"

    @patch("dslib.data_utils.urllib.request.urlretrieve", side_effect=_fake_download)
    def test_creates_parent_directories(self, mock_urlretrieve, tmp_path):
        deep_dest = tmp_path / "a" / "b" / "c" / "file.csv"

        result = download_if_missing("https://example.com/data.csv", deep_dest)

        assert result == deep_dest
        assert deep_dest.exists()


# --- Fixtures for stratified_split ---


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a binary classification parquet file (1000 rows, 70/30 split)."""
    rng = np.random.default_rng(42)
    n = 1000
    target = np.array([0] * 700 + [1] * 300)
    rng.shuffle(target)
    df = pl.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "label": target,
    })
    path = tmp_path / "binary.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def multiclass_parquet(tmp_path):
    """Create a multiclass parquet file (900 rows, 3 equal classes)."""
    rng = np.random.default_rng(99)
    n = 900
    target = np.array([0] * 300 + [1] * 300 + [2] * 300)
    rng.shuffle(target)
    df = pl.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "class": target,
    })
    path = tmp_path / "multiclass.parquet"
    df.write_parquet(path)
    return path


FEATURES = ["feat_a", "feat_b"]


class TestStratifiedSplit:
    """Tests for stratified_split."""

    def test_returns_three_splits(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=FEATURES)
        assert set(result.keys()) == {"train", "calibration", "test"}
        for key in ("train", "calibration", "test"):
            X, y = result[key]
            assert len(X) == len(y)

    def test_split_sizes_sum_to_total(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=FEATURES)
        total = sum(len(result[k][1]) for k in ("train", "calibration", "test"))
        assert total == 1000

    def test_no_index_overlap(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=FEATURES)
        train_idx = set(result["train"][0].index)
        cal_idx = set(result["calibration"][0].index)
        test_idx = set(result["test"][0].index)
        assert train_idx.isdisjoint(cal_idx)
        assert train_idx.isdisjoint(test_idx)
        assert cal_idx.isdisjoint(test_idx)

    def test_stratification_preserves_class_ratio(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=FEATURES)
        original_rate = 0.3  # 300/1000
        for key in ("train", "calibration", "test"):
            _, y = result[key]
            split_rate = y.mean()
            assert abs(split_rate - original_rate) < 0.05, (
                f"{key} rate {split_rate:.3f} deviates from {original_rate}"
            )

    def test_reproducibility_same_seed(self, sample_parquet):
        r1 = stratified_split(sample_parquet, target="label", features=FEATURES, seed=42)
        r2 = stratified_split(sample_parquet, target="label", features=FEATURES, seed=42)
        for key in ("train", "calibration", "test"):
            pd.testing.assert_frame_equal(r1[key][0], r2[key][0])
            pd.testing.assert_series_equal(r1[key][1], r2[key][1])

    def test_different_seed_gives_different_splits(self, sample_parquet):
        r1 = stratified_split(sample_parquet, target="label", features=FEATURES, seed=42)
        r2 = stratified_split(sample_parquet, target="label", features=FEATURES, seed=99)
        # At least one split should differ
        assert not r1["train"][0].equals(r2["train"][0])

    def test_returns_pandas_types(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=FEATURES)
        for key in ("train", "calibration", "test"):
            X, y = result[key]
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)

    def test_only_requested_features_in_x(self, sample_parquet):
        result = stratified_split(sample_parquet, target="label", features=["feat_a"])
        for key in ("train", "calibration", "test"):
            X, _ = result[key]
            assert list(X.columns) == ["feat_a"]

    def test_invalid_sizes_raise_value_error(self, sample_parquet):
        with pytest.raises(ValueError, match="must be < 1.0"):
            stratified_split(
                sample_parquet, target="label", features=FEATURES,
                train_size=0.7, calibration_size=0.4,
            )

    def test_multiclass_preserves_all_classes(self, multiclass_parquet):
        result = stratified_split(multiclass_parquet, target="class", features=FEATURES)
        for key in ("train", "calibration", "test"):
            _, y = result[key]
            assert set(y.unique()) == {0, 1, 2}

    def test_metadata_keys_present(self, sample_parquet):
        result = stratified_split(
            sample_parquet, target="label", features=FEATURES, include_metadata=True
        )
        assert "metadata" in result
        for key in ("train", "calibration", "test"):
            meta = result["metadata"][key]
            assert "n_samples" in meta
            assert "target_rate" in meta
            assert "pct_total" in meta
