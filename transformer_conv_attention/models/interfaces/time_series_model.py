# transformer_conv_attention/models/interfaces/time_series_model.py
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional, Dict


class TimeSeriesModel(ABC):
    """Interface for time series models"""

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            target_len: int,
            src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for time series prediction

        Args:
            x: Input sequence [batch_size, seq_length, features]
            target_len: Number of time steps to predict
            src_mask: Optional mask for input sequence

        Returns:
            - Predictions [batch_size, target_len, features]
            - Attention weights dictionary
        """
        pass

    @abstractmethod
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get attention maps from last forward pass"""
        pass