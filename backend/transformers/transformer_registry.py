from typing import Type
import logging
from .local_transformer import LocalTransformer
from .base_transformer import BaseTransformer


logger = logging.getLogger(__name__)

TRANSFORMER_REGISTRY: dict[str, Type[BaseTransformer]] = {
    "local": LocalTransformer,

}
