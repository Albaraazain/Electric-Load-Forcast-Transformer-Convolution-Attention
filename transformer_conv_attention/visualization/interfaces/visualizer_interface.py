# transformer_conv_attention/visualization/interfaces/visualizer_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

class VisualizerInterface(ABC):
    """Interface for visualization components"""

    @abstractmethod
    def plot(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create visualization"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get visualizer name"""
        pass

    @abstractmethod
    def save(self, fig: plt.Figure, path: str) -> None:
        """Save visualization"""
        pass