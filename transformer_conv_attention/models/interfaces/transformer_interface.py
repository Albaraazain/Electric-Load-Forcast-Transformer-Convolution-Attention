# transformer_conv_attention/models/interfaces/transformer_interface.py
from abc import ABC, abstractmethod
import torch
from typing import Optional

class TransformerBase(ABC):
    """Base interface for transformer models"""

    @abstractmethod
    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer

        Args:
            src: Source sequence
            tgt: Target sequence
            src_mask: Source mask
            tgt_mask: Target mask
            memory_mask: Memory mask

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def get_attention_weights(self) -> dict:
        """Get attention weights from last forward pass"""
        pass