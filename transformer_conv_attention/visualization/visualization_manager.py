# transformer_conv_attention/visualization/visualization_manager.py
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from .factory.visualizer_factory import VisualizerFactory
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VisualizationManager:
    """Manager for handling visualizations"""

    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizers = {}

    def add_visualizer(self, name: str, visualizer_type: str, **kwargs):
        """Add visualizer to manager"""
        try:
            visualizer = VisualizerFactory.create_visualizer(visualizer_type, **kwargs)
            self.visualizers[name] = visualizer
            logger.info(f"Added visualizer: {name} ({visualizer.get_name()})")
        except Exception as e:
            logger.error(f"Failed to add visualizer: {str(e)}")
            raise

    def create_visualization(
            self,
            name: str,
            data: Dict[str, Any],
            save: bool = True,
            filename: Optional[str] = None,
            **kwargs
    ):
        """Create visualization"""
        visualizer = self.visualizers.get(name)
        if visualizer is None:
            raise ValueError(f"No visualizer found with name: {name}")

        try:
            fig = visualizer.plot(data, **kwargs)

            if save:
                if filename is None:
                    filename = f"{name}_{kwargs.get('suffix', '')}.png"
                save_path = self.output_dir / filename
                visualizer.save(fig, str(save_path))

            return fig

        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise

    def create_all_visualizations(
            self,
            data: Dict[str, Any],
            save: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Create all registered visualizations"""
        results = {}
        for name in self.visualizers:
            try:
                fig = self.create_visualization(name, data, save, **kwargs)
                results[name] = fig
            except Exception as e:
                logger.error(f"Failed to create visualization {name}: {str(e)}")
                results[name] = None
        return results