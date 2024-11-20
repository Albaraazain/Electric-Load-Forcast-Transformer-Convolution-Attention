# transformer_conv_attention/evaluation/interfaces/metric_interface.py
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, Optional

class MetricInterface(ABC):
    """Interface for metrics calculation"""

    @abstractmethod
    def calculate(
            self,
            actual: Union[np.ndarray, torch.Tensor],
            predicted: Union[np.ndarray, torch.Tensor],
            mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> float:
        """Calculate metric value"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name"""
        pass