"""Contracts for Kaggle download validation and local file copies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from utils.kaggle_download import KaggleDatasetDownloader

if TYPE_CHECKING:
    from pathlib import Path


def test_empty_dataset_id_raises() -> None:
    """Construction fails when the Kaggle dataset id is empty."""
    with pytest.raises(ValueError, match="Dataset ID must be provided"):
        KaggleDatasetDownloader(dataset_id="", path_to_save="data")


def test_empty_path_raises() -> None:
    """Construction fails when the destination path is empty."""
    with pytest.raises(ValueError, match="Path to save must be provided"):
        KaggleDatasetDownloader(dataset_id="owner/name", path_to_save="")


def test_run_copies_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run() copies files from the download root into path_to_save."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")
    downloader = KaggleDatasetDownloader(
        dataset_id="owner/name", path_to_save=str(dest)
    )

    def fake_download(_data_folder: str) -> Path:
        return src

    monkeypatch.setattr(downloader, "download_dataset", fake_download)
    downloader.run()
    assert (dest / "a.txt").read_text(encoding="utf-8") == "hello"


def test_run_rejects_existing_dir_when_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dirs_exist_ok=False leaves an existing destination directory untouched."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    nested = src / "folder"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("hello", encoding="utf-8")
    (dest / "folder").mkdir(parents=True)
    downloader = KaggleDatasetDownloader(
        dataset_id="owner/name",
        path_to_save=str(dest),
        dirs_exist_ok=False,
    )

    def fake_download(_data_folder: str) -> Path:
        return src

    monkeypatch.setattr(downloader, "download_dataset", fake_download)
    with pytest.raises(FileExistsError):
        downloader.run()


def test_run_overwrites_existing_dir_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dirs_exist_ok=True copies into directories that already exist."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    nested = src / "folder"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("hello", encoding="utf-8")
    (dest / "folder").mkdir(parents=True)
    downloader = KaggleDatasetDownloader(
        dataset_id="owner/name",
        path_to_save=str(dest),
        dirs_exist_ok=True,
    )

    def fake_download(_data_folder: str) -> Path:
        return src

    monkeypatch.setattr(downloader, "download_dataset", fake_download)
    downloader.run()
    assert (dest / "folder" / "a.txt").read_text(encoding="utf-8") == "hello"
