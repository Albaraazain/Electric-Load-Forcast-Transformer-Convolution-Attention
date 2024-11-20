# transformer_conv_attention/visualization/base_visualizer.py
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from .interfaces.visualizer_interface import VisualizerInterface
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseVisualizer(VisualizerInterface):
    """Base class for visualizers"""

    def __init__(self, figsize: tuple = (10, 6), style: str = 'seaborn'):
        """
        Initialize visualizer

        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)

    def save(self, fig: plt.Figure, path: str) -> None:
        """Save figure to file"""
        try:
            fig.savefig(path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to {path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")
            raise