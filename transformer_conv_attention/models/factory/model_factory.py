# transformer_conv_attention/models/factory/model_factory.py
from typing import Dict, Type

from torch import nn

from ..interfaces.model_builder import ModelBuilder
from ...config.model_config import TransformerConfig
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ModelFactory:
    """Factory for creating transformer models"""

    _builders: Dict[str, Type[ModelBuilder]] = {}

    @classmethod
    def register_builder(
            cls,
            name: str,
            builder: Type[ModelBuilder]
    ) -> None:
        """
        Register a new model builder
        
        Args:
            name: Name of the model type
            builder: Builder class for the model
        """
        if name in cls._builders:
            logger.warning(f"Builder {name} already registered. Overwriting...")
        cls._builders[name] = builder
        logger.info(f"Registered builder: {name}")

    @classmethod
    def create_model(
            cls,
            model_type: str,
            config: TransformerConfig
    ) -> nn.Module:
        """
        Create a model using the registered builder
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Created model
            
        Raises:
            ValueError: If no builder is registered for model_type
        """
        builder_cls = cls._builders.get(model_type)
        if not builder_cls:
            raise ValueError(
                f"No builder registered for model type: {model_type}. "
                f"Available types: {list(cls._builders.keys())}"
            )

        builder = builder_cls()
        model = builder.build_model(config.to_dict())

        logger.info(f"Created model of type: {model_type}")
        return model