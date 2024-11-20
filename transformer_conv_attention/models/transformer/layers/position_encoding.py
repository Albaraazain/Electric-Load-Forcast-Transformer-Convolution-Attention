# transformer_conv_attention/models/transformer/layers/position_encoding.py
import torch
import torch.nn as nn
import math
from typing import Optional
from ....utils.logger import get_logger

logger = get_logger(__name__)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding

        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin/cos positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (won't be updated during training)
        self.register_buffer('pe', pe)

        logger.info(
            f"Initialized positional encoding with d_model={d_model}, "
            f"max_seq_length={max_seq_length}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return self.dropout(x + self.pe[:, :x.size(1)])