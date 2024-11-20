# transformer_conv_attention/visualization/visualizers/prediction_visualizer.py
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from ..base_visualizer import BaseVisualizer

class PredictionVisualizer(BaseVisualizer):
    """Visualizer for predictions vs actual values"""

    def plot(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        timestamps = data['timestamps']
        actual = data['actual']
        predicted = data['predicted']
        prediction_horizons = kwargs.get('horizons', [24, 48, 96])

        fig, axes = plt.subplots(
            len(prediction_horizons), 1,
            figsize=(self.figsize[0], self.figsize[1] * len(prediction_horizons)),
            sharex=True
        )

        if len(prediction_horizons) == 1:
            axes = [axes]

        for ax, horizon in zip(axes, prediction_horizons):
            ax.plot(timestamps, actual, label='Actual', color='black', alpha=0.7)
            ax.plot(timestamps, predicted[:, horizon-1],
                    label=f'{horizon}h Prediction', alpha=0.7)
            ax.set_title(f'{horizon}-hour Ahead Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_name(self) -> str:
        return "Prediction Plot"