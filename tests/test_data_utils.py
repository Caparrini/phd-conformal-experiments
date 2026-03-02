"""Tests for dslib.data_utils."""

import hashlib
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from dslib.data_utils import download_if_missing


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
    def test_raises_on_hash_mismatch_after_download(
        self, mock_urlretrieve, tmp_dest
    ):
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