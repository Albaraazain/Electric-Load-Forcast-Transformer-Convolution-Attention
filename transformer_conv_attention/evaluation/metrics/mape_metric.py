# transformer_conv_attention/evaluation/metrics/mape_metric.py
from .base_metric import BaseMetric
import numpy as np

class MAPEMetric(BaseMetric):
    """Mean Absolute Percentage Error"""

    def calculate(self, actual, predicted, mask=None):
        actual, predicted = self._prepare_data(actual, predicted, mask)
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def get_name(self) -> str:
        return "MAPE"
