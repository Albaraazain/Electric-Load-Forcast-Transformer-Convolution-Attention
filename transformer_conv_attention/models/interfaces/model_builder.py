# transformer_conv_attention/models/interfaces/model_builder.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn

class ModelBuilder(ABC):
    """Interface for model builders"""

    @abstractmethod
    def build_attention(self, config: Dict[str, Any]) -> nn.Module:
        """Build attention mechanism"""
        pass

    @abstractmethod
    def build_encoder_layer(self, config: Dict[str, Any]) -> nn.Module:
        """Build encoder layer"""
        pass

    @abstractmethod
    def build_decoder_layer(self, config: Dict[str, Any]) -> nn.Module:
        """Build decoder layer"""
        pass

    @abstractmethod
    def build_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build full encoder"""
        pass

    @abstractmethod
    def build_decoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build full decoder"""
        pass

    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build complete model"""
        pass