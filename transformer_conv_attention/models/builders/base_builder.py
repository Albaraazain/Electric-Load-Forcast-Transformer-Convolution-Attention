from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn

class ModelBuilder(ABC):
    """Base class for model builders"""

    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build complete model"""
        pass