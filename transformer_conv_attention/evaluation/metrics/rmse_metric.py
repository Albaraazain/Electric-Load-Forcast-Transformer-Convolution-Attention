import numpy as np
from .base_metric import BaseMetric

class RMSEMetric(BaseMetric):
    """Root Mean Square Error"""

    def compute(self, actual, predicted, mask=None):
        actual, predicted = self._prepare_data(actual, predicted, mask)
        return np.sqrt(np.mean((actual - predicted) ** 2))

    def get_name(self) -> str:
        return "RMSE"