import numpy as np
from .base_metric import BaseMetric

class MAPEMetric(BaseMetric):
    """Mean Absolute Percentage Error"""

    def compute(self, actual, predicted, mask=None):
        actual, predicted = self._prepare_data(actual, predicted, mask)
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def get_name(self) -> str:
        return "MAPE"