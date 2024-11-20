# transformer_conv_attention/models/transformer/layers/encoder_layer.py
import torch
import torch.nn as nn
from typing import Optional
from ..layers.conv_attention import ConvolutionalAttention
from ....utils.logger import get_logger

logger = get_logger(__name__)

class EncoderLayer(nn.Module):
    """Transformer encoder layer with convolutional attention"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            kernel_size: int,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5
    ):
        """
        Initialize encoder layer

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feedforward network dimension
            kernel_size: Kernel size for convolutional attention
            dropout: Dropout rate
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()

        # Self attention
        self.self_attn = ConvolutionalAttention(
            d_model=d_model,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Feedforward network
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized encoder layer with d_model={d_model}, "
            f"n_heads={n_heads}, d_ff={d_ff}"
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer

        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Self attention
        attn_out, self.attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=mask
        )

        # First residual connection
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward network
        ff_out = self.ff_net(x)

        # Second residual connection
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x