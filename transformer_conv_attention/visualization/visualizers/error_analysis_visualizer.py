# transformer_conv_attention/visualization/visualizers/error_analysis_visualizer.py
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from transformer_conv_attention.visualization.base_visualizer import BaseVisualizer




class ErrorAnalysisVisualizer(BaseVisualizer):
    """Visualizer for error analysis"""

    def plot(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        actual = data['actual']
        predicted = data['predicted']
        timestamps = data['timestamps']

        errors = predicted - actual

        fig = plt.figure(figsize=(self.figsize[0] * 2, self.figsize[1] * 2))

        # Create subplots
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # Error over time
        ax2 = fig.add_subplot(gs[0, 1])  # Error distribution
        ax3 = fig.add_subplot(gs[1, :])  # Error vs actual value

        # Error over time
        ax1.plot(timestamps, errors, alpha=0.7)
        ax1.set_title('Prediction Error Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Error')
        ax1.grid(True, alpha=0.3)

        # Error distribution
        ax2.hist(errors, bins=50, alpha=0.7)
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Error vs actual value
        ax3.scatter(actual, errors, alpha=0.5)
        ax3.set_title('Error vs Actual Value')
        ax3.set_xlabel('Actual Value')
        ax3.set_ylabel('Error')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_name(self) -> str:
        return "Error Analysis"

