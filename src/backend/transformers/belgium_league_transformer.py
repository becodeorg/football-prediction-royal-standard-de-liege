import logging
from src.backend.transformers.local_transformer import LocalTransformer

logger = logging.getLogger(__name__)


class BelgiumLeagueTransformer(LocalTransformer):
    """
    Uses base LocalTransformer logic for Belgium league.
    """
    ...
