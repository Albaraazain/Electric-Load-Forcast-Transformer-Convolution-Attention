# transformer_conv_attention/models/interfaces/attention_interface.py
from abc import ABC, abstractmethod
import torch
from typing import Optional, Tuple

class AttentionMechanism(ABC):
    """Base interface for attention mechanisms"""

    @abstractmethod
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            - Output tensor
            - Attention weights (optional)
        """
        pass