from abc import ABC, abstractmethod
from typing import Dict, Type

from models.interfaces.model_builder import ModelBuilder


class ModelFactory:
    """Factory for creating transformer models"""

    _builders: Dict[str, Type['ModelBuilder']] = {}

    @classmethod
    def register_builder(cls, model_type: str, builder: Type['ModelBuilder']):
        cls._builders[model_type] = builder

    @classmethod
    def create_model(cls, model_type: str, config: dict):
        builder = cls._builders.get(model_type)
        if not builder:
            raise ValueError(f"No builder registered for {model_type}")
        return builder().build(config)