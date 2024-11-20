# transformer_conv_attention/visualization/factory/visualizer_factory.py
from typing import Dict, Type
from ..interfaces.visualizer_interface import VisualizerInterface
from ..visualizers.prediction_visualizer import PredictionVisualizer
from ..visualizers.attention_visualizer import AttentionVisualizer
from ..visualizers.error_analysis_visualizer import ErrorAnalysisVisualizer

class VisualizerFactory:
    """Factory for creating visualizers"""

    _visualizers: Dict[str, Type[VisualizerInterface]] = {
        'prediction': PredictionVisualizer,
        'attention': AttentionVisualizer,
        'error': ErrorAnalysisVisualizer
    }

    @classmethod
    def create_visualizer(cls, visualizer_type: str, **kwargs) -> VisualizerInterface:
        """Create visualizer instance"""
        visualizer_class = cls._visualizers.get(visualizer_type.lower())
        if visualizer_class is None:
            raise ValueError(f"Unknown visualizer type: {visualizer_type}")
        return visualizer_class(**kwargs)

    @classmethod
    def register_visualizer(cls, name: str, visualizer_class: Type[VisualizerInterface]):
        """Register new visualizer"""
        cls._visualizers[name.lower()] = visualizer_class