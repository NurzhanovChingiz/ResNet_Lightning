from utils.kaggle_download import KaggleDatasetDownloader
from config import CFG


if __name__ == "__main__":
    downloader = KaggleDatasetDownloader(dataset_id=CFG.DATASET_ID, path_to_save=CFG.DOWNLOAD_PATH)
    downloader.run()