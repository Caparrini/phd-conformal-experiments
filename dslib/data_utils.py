"""Utilities for downloading and managing datasets."""

import hashlib
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError

from loguru import logger


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
