# transformer_conv_attention/evaluation/metrics/rmse_metric.py
from .base_metric import BaseMetric
import numpy as np

class RMSEMetric(BaseMetric):
    """Root Mean Square Error"""

    def calculate(self, actual, predicted, mask=None):
        actual, predicted = self._prepare_data(actual, predicted, mask)
        return np.sqrt(np.mean((actual - predicted) ** 2))

    def get_name(self) -> str:
        return "RMSE"