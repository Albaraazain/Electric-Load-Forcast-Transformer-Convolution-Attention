# transformer_conv_attention/models/registry.py
from typing import Dict, Type
from .factory.model_factory import ModelFactory
from .builders.time_series_builder import TimeSeriesTransformerBuilder
from .builders.base_builder import ModelBuilder
from ..utils.logger import get_logger

logger = get_logger(__name__)

_BUILDERS = {
    'time_series_transformer': TimeSeriesTransformerBuilder,
}

def register_builders() -> None:
    """Register all model builders"""
    logger.info("Registering model builders...")
    for name, builder in _BUILDERS.items():
        logger.info(f"Registering builder: {name}")
        ModelFactory.register_builder(name, builder)

def get_available_models() -> list:
    """Get list of available model types"""
    return list(_BUILDERS.keys())