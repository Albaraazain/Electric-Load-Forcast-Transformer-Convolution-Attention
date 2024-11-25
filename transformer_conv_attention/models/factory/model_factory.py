from typing import Dict, Type, Any
import torch.nn as nn
from ..builders.base_builder import ModelBuilder
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ModelFactory:
    """Factory for creating transformer models"""

    _builders: Dict[str, Type[ModelBuilder]] = {}

    @classmethod
    def register_builder(cls, name: str, builder: Type[ModelBuilder]) -> None:
        """Register a new model builder"""
        if name in cls._builders:
            logger.warning(f"Builder {name} already registered. Overwriting...")
        cls._builders[name] = builder
        logger.info(f"Registered builder: {name}")

    @classmethod
    def create_model(cls, model_type: str, config: Any) -> nn.Module:
        """Create a model using the registered builder"""
        print(f"[DEBUG] Available builders: {list(cls._builders.keys())}")
        builder_cls = cls._builders.get(model_type)
        if not builder_cls:
            raise ValueError(
                f"No builder registered for model type: {model_type}. "
                f"Available types: {list(cls._builders.keys())}"
            )

        builder = builder_cls()
        return builder.build_model(config)