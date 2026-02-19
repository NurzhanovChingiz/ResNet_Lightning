import kagglehub  # type: ignore[import-untyped]
import shutil
from pathlib import Path

class KaggleDatasetDownloader:
    def __init__(self, dataset_id: str, path_to_save: str, dirs_exist_ok: bool = False) -> None:
        '''
        Initializes the KaggleDatasetDownloader with the specified dataset ID.
        Args:
            dataset_id (str): The Kaggle dataset identifier (e.g., Link to Kaggle).
        '''
        self.dataset_id = dataset_id
        assert self.dataset_id, "Dataset ID must be provided."
        assert isinstance(self.dataset_id, str), "Dataset ID must be a string."
        self.path_to_save = path_to_save
        assert self.path_to_save, "Path to save must be provided."
        assert isinstance(self.path_to_save, str), "Path to save must be a string."
        self.dirs_exist_ok = dirs_exist_ok
    
    def download_dataset(self, data_folder: str) -> Path:
        '''
        Downloads the dataset from Kaggle and returns the source root path.
        Args:
            data_folder (str): The path to the folder.
        '''
        src_root = Path(kagglehub.dataset_download(self.dataset_id))
        return src_root

    def _move_items(self, src_root: Path, data_folder_path: Path, dirs_exist_ok: bool = False) -> None:
        """
        Copies all items from src_root to data_folder_path.

        Args:
            src_root (Path): Source root path.
            data_folder_path (Path): Destination folder path.
            dirs_exist_ok (bool): If True, allows existing directories to be overwritten.
        """

        # Ensure destination directory exists
        data_folder_path.mkdir(parents=True, exist_ok=True)

        # Iterate through all items in the source directory
        for item in src_root.iterdir():
            dst = data_folder_path / item.name

            if item.is_dir():
                # Copy entire directory tree
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                # Copy individual file (will overwrite if exists)
                shutil.copy2(item, dst)

    
    def run(self) -> None:
        '''
        Executes the dataset download and organization process.
        '''
        print(f"Downloading {self.dataset_id} dataset to:", self.path_to_save)
        src_root = self.download_dataset(self.path_to_save)
        data_folder_path = Path(self.path_to_save)
        self._move_items(src_root, data_folder_path, dirs_exist_ok=self.dirs_exist_ok)
        print(f"{self.dataset_id} dataset downloaded and organized.")


 