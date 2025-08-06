from typing import Type
import logging

from src.backend.transformers.local_transformer import LocalTransformer
from src.backend.transformers.belgium_league_transformer import BelgiumLeagueTransformer
from src.backend.transformers.base_transformer import BaseTransformer

logger = logging.getLogger(__name__)

TRANSFORMER_REGISTRY: dict[str, Type[BaseTransformer]] = {
    "B1_old": LocalTransformer,
    "Belgium_league_2526": BelgiumLeagueTransformer

}
