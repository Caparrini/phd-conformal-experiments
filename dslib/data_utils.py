"""Utilities for downloading and managing datasets."""

import hashlib
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError

from loguru import logger
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


def _verify_hash(path: Path, expected_sha256: str) -> bool:
    """Verify a file's SHA-256 hash against an expected value."""
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != expected_sha256:
        logger.warning(
            f"Hash mismatch for {path.name}: "
            f"expected {expected_sha256[:12]}..., got {file_hash[:12]}..."
        )
        return False
    logger.info(f"Hash verified: {expected_sha256[:12]}...")
    return True


def download_if_missing(url: str, dest: str | Path, sha256: str | None = None) -> Path:
    """Download a file from a URL if it doesn't already exist locally.

    If the file exists and a sha256 is provided, it verifies the hash.
    If the hash doesn't match, the file is re-downloaded.

    Args:
        url: Direct download URL.
        dest: Local destination path.
        sha256: Optional SHA-256 hash to verify the file.

    Returns:
        Path to the verified file.

    Raises:
        HTTPError: If the server returns an error (404, 500, etc.).
        URLError: If the URL is unreachable (DNS failure, no connection, etc.).
        ValueError: If the downloaded file's hash doesn't match after download.
    """
    dest = Path(dest)

    if dest.exists():
        if sha256 and not _verify_hash(dest, sha256):
            logger.info(f"Re-downloading {dest.name}...")
            dest.unlink()
        else:
            logger.info(f"File already exists: {dest}")
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading {dest.name} from {url}...")
        urllib.request.urlretrieve(url, dest)
    except HTTPError as e:
        logger.error(f"Server error downloading {url}: {e.code} {e.reason}")
        raise
    except URLError as e:
        logger.error(f"Could not reach {url}: {e.reason}")
        raise

    if sha256 and not _verify_hash(dest, sha256):
        dest.unlink()
        raise ValueError(
            f"Hash mismatch for {dest.name} after download. "
            f"The remote file may have changed."
        )

    size_mb = dest.stat().st_size / 1024 / 1024
    logger.info(f"Downloaded: {dest} ({size_mb:.1f} MB)")
    return dest


def load_and_split(
    data_path: str | Path,
    date_column: str,
    train_cutoff: str,
    calibration_cutoff: str,
    target: str,
    features: list[str],
    include_metadata: bool = False,
) -> dict:
    """
    Load parquet and split into train/calibration/test by date.

    Parameters
    ----------
    data_path : path to parquet file
    date_column : column name with dates
    train_cutoff : ISO date string for train/cal boundary
    calibration_cutoff : ISO date string for cal/test boundary
    target : target column name
    features : list of feature column names
    include_metadata : if True, include split statistics

    Returns
    -------
    dict with 'train', 'calibration', 'test' as (X, y) tuples.
    If include_metadata=True, also includes 'metadata' with split stats.
    """
    from datetime import date
    from pathlib import Path
    import polars as pl

    df = pl.read_parquet(Path(data_path))
    train_cutoff_date = date.fromisoformat(train_cutoff)
    cal_cutoff_date = date.fromisoformat(calibration_cutoff)

    train_data = df.filter(pl.col(date_column) <= train_cutoff_date)
    cal_data = df.filter(
        (pl.col(date_column) > train_cutoff_date)
        & (pl.col(date_column) <= cal_cutoff_date)
    )
    test_data = df.filter(pl.col(date_column) > cal_cutoff_date)

    def _to_xy(data):
        X = data.select(features).to_pandas()
        y = data[target].to_pandas()
        return X, y

    result = {
        "train": _to_xy(train_data),
        "calibration": _to_xy(cal_data),
        "test": _to_xy(test_data),
    }

    if include_metadata:
        total = df.shape[0]
        result["metadata"] = {}
        for name, data in [("train", train_data), ("calibration", cal_data), ("test", test_data)]:
            result["metadata"][name] = {
                "n_samples": data.shape[0],
                "default_rate": float(data[target].mean()),
                "date_from": str(data[date_column].min()),
                "date_to": str(data[date_column].max()),
                "pct_total": data.shape[0] / total,
            }

    return result


def stratified_split(
    data_path: str | Path,
    target: str,
    features: list[str],
    train_size: float = 0.6,
    calibration_size: float = 0.2,
    seed: int = 42,
    include_metadata: bool = False,
) -> dict:
    """
    Load parquet and split into train/calibration/test with stratification.

    Parameters
    ----------
    data_path : str or Path
        Path to parquet file.
    target : str
        Target column name.
    features : list of str
        List of feature column names.
    train_size : float
        Fraction of data for training (default 0.6).
    calibration_size : float
        Fraction of data for calibration (default 0.2).
    seed : int
        Random seed for reproducibility.
    include_metadata : bool
        If True, include split statistics in the result.

    Returns
    -------
    dict
        Keys 'train', 'calibration', 'test' as (X, y) tuples of pandas
        DataFrame/Series. If include_metadata=True, also includes 'metadata'.

    Raises
    ------
    ValueError
        If train_size + calibration_size >= 1.0.
    """
    if train_size + calibration_size >= 1.0:
        raise ValueError(
            f"train_size + calibration_size must be < 1.0, "
            f"got {train_size} + {calibration_size} = {train_size + calibration_size}"
        )

    df = pl.read_parquet(Path(data_path))
    X_all = df.select(features).to_pandas()
    y_all = df[target].to_pandas()

    # First split: train vs (calibration + test)
    rest_size = 1.0 - train_size
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_all, y_all, test_size=rest_size, stratify=y_all, random_state=seed
    )

    # Second split: calibration vs test from the rest
    cal_fraction = calibration_size / rest_size
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=1.0 - cal_fraction, stratify=y_rest, random_state=seed
    )

    result = {
        "train": (X_train, y_train),
        "calibration": (X_cal, y_cal),
        "test": (X_test, y_test),
    }

    if include_metadata:
        total = len(y_all)
        n_unique = y_all.nunique()
        result["metadata"] = {}
        for name, y_split in [("train", y_train), ("calibration", y_cal), ("test", y_test)]:
            if n_unique <= 2:
                target_rate = float(np.mean(y_split))
            else:
                target_rate = y_split.value_counts(normalize=True).to_dict()
            result["metadata"][name] = {
                "n_samples": len(y_split),
                "target_rate": target_rate,
                "pct_total": len(y_split) / total,
            }

    return result
