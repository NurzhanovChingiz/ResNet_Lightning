"""Download the configured Kaggle dataset into CFG.DOWNLOAD_PATH."""

from config import CFG
from utils.kaggle_download import KaggleDatasetDownloader

if __name__ == "__main__":
    downloader = KaggleDatasetDownloader(
        dataset_id=CFG.DATASET_ID, path_to_save=CFG.DOWNLOAD_PATH
    )
    downloader.run()
