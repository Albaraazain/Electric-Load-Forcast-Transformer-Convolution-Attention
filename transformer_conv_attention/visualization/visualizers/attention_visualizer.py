# transformer_conv_attention/visualization/visualizers/attention_visualizer.py
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from transformer_conv_attention.visualization.base_visualizer import BaseVisualizer


class AttentionVisualizer(BaseVisualizer):
    """Visualizer for attention weights"""

    def plot(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        attention_weights = data['attention_weights']
        timestamps = data['timestamps']
        layer_index = kwargs.get('layer_index', 0)
        head_index = kwargs.get('head_index', 0)

        fig, ax = plt.subplots(figsize=self.figsize)

        weights = attention_weights[layer_index][head_index]
        im = ax.imshow(weights, aspect='auto', cmap='viridis')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Set labels
        ax.set_title(f'Attention Weights (Layer {layer_index}, Head {head_index})')
        ax.set_xlabel('Input Time Steps')
        ax.set_ylabel('Output Time Steps')

        # Add timestamp ticks
        if len(timestamps) > 10:
            tick_indices = np.linspace(0, len(timestamps)-1, 10, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([timestamps[i].strftime('%Y-%m-%d %H:%M')
                                for i in tick_indices], rotation=45)

        plt.tight_layout()
        return fig

    def get_name(self) -> str:
        return "Attention Heatmap"