import logging
from typing import Type
from src.backend.downloaders.local_downloader import LocalDownloader
from src.backend.downloaders.belgium_league_downloader import BelgiumLeagueDownloader
from src.backend.downloaders.base_downloder import BaseDownloader

logger = logging.getLogger(__name__)

# Registry mapping downloader names to their corresponding classes
DOWNLOADER_REGISTRY: dict[str, Type[BaseDownloader]] = {
    "local": LocalDownloader,
    "Belgium_league": BelgiumLeagueDownloader

}
