"""Download a Kaggle dataset and copy it into a local folder."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

LOGGER = logging.getLogger(__name__)


class KaggleDatasetDownloader:
    """Download one Kaggle dataset and copy its files to a destination folder."""

    def __init__(
        self, dataset_id: str, path_to_save: str, *, dirs_exist_ok: bool = False
    ) -> None:
        """Store the Kaggle dataset id and the local destination path.

        Args:
            dataset_id: The Kaggle dataset identifier.
            path_to_save: Local directory that should receive the files.
            dirs_exist_ok: If True, copy into directories that already exist.
        """
        if not dataset_id:
            message = "Dataset ID must be provided."
            raise ValueError(message)
        if not isinstance(dataset_id, str):
            message = "Dataset ID must be a string."
            raise TypeError(message)
        if not path_to_save:
            message = "Path to save must be provided."
            raise ValueError(message)
        if not isinstance(path_to_save, str):
            message = "Path to save must be a string."
            raise TypeError(message)
        self.dataset_id = dataset_id
        self.path_to_save = path_to_save
        self.dirs_exist_ok = dirs_exist_ok

    def download_dataset(self, _data_folder: str) -> Path:
        """Download the dataset through kagglehub and return its cache path.

        Args:
            _data_folder: Destination folder accepted for caller compatibility.

        Returns:
            Path to the kagglehub download root.
        """
        return Path(kagglehub.dataset_download(self.dataset_id))

    def _move_items(
        self,
        src_root: Path,
        data_folder_path: Path,
        *,
        dirs_exist_ok: bool = False,
    ) -> None:
        """Copy every item from ``src_root`` into ``data_folder_path``.

        Args:
            src_root: Source root path.
            data_folder_path: Destination folder path.
            dirs_exist_ok: If True, allow existing destination directories.
        """
        data_folder_path.mkdir(parents=True, exist_ok=True)

        for item in src_root.iterdir():
            destination = data_folder_path / item.name

            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=dirs_exist_ok)
            else:
                shutil.copy2(item, destination)

    def run(self) -> None:
        """Download the dataset and copy it into ``path_to_save``.

        Strategy:
            Obtain the kagglehub cache root for this dataset, then copy every
            item into ``path_to_save``.

        Flow:
            1. Download cache root via download_dataset
            2. Copy into path_to_save via _move_items
        """
        LOGGER.info("Downloading %s dataset to: %s", self.dataset_id, self.path_to_save)
        # 1. Download cache root via download_dataset
        src_root = self.download_dataset(self.path_to_save)
        # 2. Copy into path_to_save via _move_items
        data_folder_path = Path(self.path_to_save)
        self._move_items(src_root, data_folder_path, dirs_exist_ok=self.dirs_exist_ok)
        LOGGER.info("%s dataset downloaded and organized.", self.dataset_id)
