import logging
from typing import Type
from .local_downloader import LocalDownloader
from .base_downloder import BaseDownloader

logger = logging.getLogger(__name__)

# Registry mapping downloader names to their corresponding classes
DOWNLOADER_REGISTRY: dict[str, Type[BaseDownloader]] = {
    "local": LocalDownloader

}
