# transformer_conv_attention/evaluation/interfaces/evaluator_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

class EvaluatorInterface(ABC):
    """Interface for model evaluation"""

    @abstractmethod
    def evaluate(
            self,
            dataloader: torch.utils.data.DataLoader,
            prediction_length: int
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate model performance"""
        pass

    @abstractmethod
    def add_metric(self, metric: str) -> None:
        """Add a metric for evaluation"""
        pass

