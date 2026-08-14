import importlib.util
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "hyperparams"
    / "get_hyperparams.py"
)
_spec = importlib.util.spec_from_file_location("get_hyperparams", _MODULE_PATH)
get_hyperparams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(get_hyperparams)


def _fake_zenodo_response(files: list[dict]) -> io.BytesIO:
    payload = json.dumps({"files": files}).encode()
    return io.BytesIO(payload)


# --- download_from_zenodo -------------------------------------------------


def test_download_from_zenodo_skips_if_dest_exists(tmp_path):
    dest = tmp_path / "already_here.h5"
    dest.write_bytes(b"existing content")

    with patch("urllib.request.urlopen") as mock_urlopen:
        result = get_hyperparams.download_from_zenodo("123", "some.h5", dest)

    mock_urlopen.assert_not_called()
    assert result == dest
    assert dest.read_bytes() == b"existing content"


def test_download_from_zenodo_downloads_matching_file(tmp_path):
    dest = tmp_path / "target.h5"
    cached = tmp_path / "cached_download.h5"
    cached.write_bytes(b"real content")

    files = [
        {
            "key": "target.h5",
            "size": len(b"real content"),
            "links": {"self": "https://example.org/target.h5"},
        }
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with patch.object(get_hyperparams, "download_file", return_value=str(cached)):
            result = get_hyperparams.download_from_zenodo("123", "target.h5", dest)

    assert result == dest
    assert dest.read_bytes() == b"real content"


def test_download_from_zenodo_raises_if_file_not_in_record(tmp_path):
    dest = tmp_path / "missing.h5"
    files = [
        {"key": "other_file.h5", "size": 1, "links": {"self": "https://example.org/x"}}
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with pytest.raises(FileNotFoundError, match="missing.h5"):
            get_hyperparams.download_from_zenodo("123", "missing.h5", dest)

    assert not dest.exists()


def test_download_from_zenodo_raises_and_deletes_on_size_mismatch(tmp_path):
    """Guards against the real bug hit in production: an interrupted
    download left a truncated file that downstream code silently accepted
    as valid (h5py could open it, but a required group was missing)."""
    dest = tmp_path / "target.h5"
    cached = tmp_path / "cached_download.h5"
    cached.write_bytes(b"truncated")  # 9 bytes, but record says 12345

    files = [
        {
            "key": "target.h5",
            "size": 12345,
            "links": {"self": "https://example.org/target.h5"},
        }
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with patch.object(get_hyperparams, "download_file", return_value=str(cached)):
            with pytest.raises(OSError, match="Incomplete download"):
                get_hyperparams.download_from_zenodo("123", "target.h5", dest)

    assert not dest.exists()
    assert not cached.exists()  # truncated cached file removed, not left behind


# --- download_from_dcc -----------------------------------------------------


def _fake_head_response(content_length: int):
    response = MagicMock()
    response.headers = {"Content-Length": str(content_length)}
    response.__enter__ = lambda self: response
    response.__exit__ = lambda self, *args: False
    return response


def test_download_from_dcc_skips_if_dest_exists(tmp_path):
    dest = tmp_path / "already_here.hdf5"
    dest.write_bytes(b"existing")

    with patch.object(get_hyperparams, "download_file") as mock_download:
        result = get_hyperparams.download_from_dcc("some_file.hdf5", dest)

    mock_download.assert_not_called()
    assert result == dest


def test_download_from_dcc_downloads_and_moves(tmp_path):
    dest = tmp_path / "target.hdf5"
    cached = tmp_path / "cached.hdf5"
    cached.write_bytes(b"dcc content")

    with patch(
        "urllib.request.urlopen", return_value=_fake_head_response(len(b"dcc content"))
    ):
        with patch.object(
            get_hyperparams, "download_file", return_value=str(cached)
        ) as mock_download:
            result = get_hyperparams.download_from_dcc("baseline5_result.hdf5", dest)

    mock_download.assert_called_once()
    called_url = mock_download.call_args[0][0]
    assert called_url == f"{get_hyperparams.DCC_BASE_URL}/baseline5_result.hdf5"
    assert result == dest
    assert dest.read_bytes() == b"dcc content"


def test_download_from_dcc_raises_and_deletes_on_size_mismatch(tmp_path):
    """Reproduces the exact production bug: DCC serves Content-Length
    9658776, but an interrupted transfer left a 2536-byte file that h5py
    could still open (valid HDF5 header) yet was missing required groups."""
    dest = tmp_path / "target.hdf5"
    cached = tmp_path / "cached.hdf5"
    cached.write_bytes(b"x" * 2536)

    with patch("urllib.request.urlopen", return_value=_fake_head_response(9658776)):
        with patch.object(get_hyperparams, "download_file", return_value=str(cached)):
            with pytest.raises(OSError, match="Incomplete download"):
                get_hyperparams.download_from_dcc("baseline5_result.hdf5", dest)

    assert not dest.exists()
    assert not cached.exists()


# --- download_and_extract_from_zenodo_tarball ------------------------------


def _make_tarball(tmp_path, member_name: str, content: bytes) -> Path:
    member_path = tmp_path / "member_source"
    member_path.write_bytes(content)

    tar_path = tmp_path / "archive.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(member_path, arcname=member_name)
    return tar_path


def test_download_and_extract_from_zenodo_tarball_skips_if_dest_exists(tmp_path):
    dest = tmp_path / "already_here.h5"
    dest.write_bytes(b"existing")

    with patch("urllib.request.urlopen") as mock_urlopen:
        result = get_hyperparams.download_and_extract_from_zenodo_tarball(
            "123", "archive.tar.gz", "target.h5", dest
        )

    mock_urlopen.assert_not_called()
    assert result == dest


def test_download_and_extract_from_zenodo_tarball_extracts_matching_member(tmp_path):
    dest = tmp_path / "out" / "fullpop.h5"
    tar_path = _make_tarball(
        tmp_path, "popsummary_files/production_1_mass_target.h5", b"popsummary bytes"
    )

    files = [
        {
            "key": "archive.tar.gz",
            "size": tar_path.stat().st_size,
            "links": {"self": "https://example.org/archive.tar.gz"},
        }
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with patch.object(get_hyperparams, "download_file", return_value=str(tar_path)):
            result = get_hyperparams.download_and_extract_from_zenodo_tarball(
                "123", "archive.tar.gz", "target.h5", dest
            )

    assert result == dest
    assert dest.read_bytes() == b"popsummary bytes"


def test_download_and_extract_from_zenodo_tarball_raises_if_archive_not_in_record(
    tmp_path,
):
    dest = tmp_path / "target.h5"
    files = [
        {"key": "other.tar.gz", "size": 1, "links": {"self": "https://example.org/x"}}
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with pytest.raises(FileNotFoundError, match="archive.tar.gz"):
            get_hyperparams.download_and_extract_from_zenodo_tarball(
                "123", "archive.tar.gz", "target.h5", dest
            )


def test_download_and_extract_from_zenodo_tarball_raises_if_member_missing(tmp_path):
    dest = tmp_path / "out" / "fullpop.h5"
    tar_path = _make_tarball(tmp_path, "unrelated_file.h5", b"unrelated bytes")

    files = [
        {
            "key": "archive.tar.gz",
            "size": tar_path.stat().st_size,
            "links": {"self": "https://example.org/archive.tar.gz"},
        }
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with patch.object(get_hyperparams, "download_file", return_value=str(tar_path)):
            with pytest.raises(FileNotFoundError, match="target.h5"):
                get_hyperparams.download_and_extract_from_zenodo_tarball(
                    "123", "archive.tar.gz", "target.h5", dest
                )


def test_download_and_extract_from_zenodo_tarball_raises_and_deletes_on_size_mismatch(
    tmp_path,
):
    dest = tmp_path / "out" / "fullpop.h5"
    tar_path = _make_tarball(tmp_path, "popsummary_files/target.h5", b"some content")

    files = [
        {
            "key": "archive.tar.gz",
            "size": tar_path.stat().st_size + 999,  # wrong on purpose
            "links": {"self": "https://example.org/archive.tar.gz"},
        }
    ]

    with patch("urllib.request.urlopen", return_value=_fake_zenodo_response(files)):
        with patch.object(get_hyperparams, "download_file", return_value=str(tar_path)):
            with pytest.raises(OSError, match="Incomplete download"):
                get_hyperparams.download_and_extract_from_zenodo_tarball(
                    "123", "archive.tar.gz", "target.h5", dest
                )

    assert not dest.exists()
    assert not tar_path.exists()
