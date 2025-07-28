import logging

from config import settings

logger = logging.getLogger(__name__)


class KaggleDataDownloader:
    def __init__(self):
        self.dataset_name = settings.kaggle_dataset_name

    def download(self):
        # Download from Kaggle
        pass

    def extract(self):
        # Unpacking
        pass
