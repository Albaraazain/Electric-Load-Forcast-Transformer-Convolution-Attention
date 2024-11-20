# transformer_conv_attention/evaluation/metrics/base_metric.py
import numpy as np
import torch
from typing import Union, Optional
from ..interfaces.metric_interface import MetricInterface

class BaseMetric(MetricInterface):
    """Base class for metrics"""

    def _prepare_data(
            self,
            actual: Union[np.ndarray, torch.Tensor],
            predicted: Union[np.ndarray, torch.Tensor],
            mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> tuple:
        """Prepare data for metric calculation"""
        if torch.is_tensor(actual):
            actual = actual.numpy()
        if torch.is_tensor(predicted):
            predicted = predicted.numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.numpy()

        if mask is not None:
            actual = actual[mask]
            predicted = predicted[mask]

        return actual, predicted


