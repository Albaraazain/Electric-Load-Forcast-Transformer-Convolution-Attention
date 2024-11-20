# transformer_conv_attention/models/transformer/layers/decoder_layer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.conv_attention import ConvolutionalAttention
from ....utils.logger import get_logger

logger = get_logger(__name__)

class DecoderLayer(nn.Module):
    """Transformer decoder layer with convolutional attention"""

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
        Initialize decoder layer

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feedforward network dimension
            kernel_size: Kernel size for convolutional attention
            dropout: Dropout rate
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()

        # Self and cross attention
        self.self_attn = ConvolutionalAttention(
            d_model=d_model,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.cross_attn = ConvolutionalAttention(
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
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized decoder layer with d_model={d_model}, "
            f"n_heads={n_heads}, d_ff={d_ff}"
        )

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer

        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target sequence mask
            memory_mask: Memory mask for cross attention

        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Self attention
        self_attn_out, self.self_attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask
        )

        # First residual connection
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # Cross attention
        cross_attn_out, self.cross_attn_weights = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            mask=memory_mask
        )

        # Second residual connection
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        # Feedforward network
        ff_out = self.ff_net(x)

        # Third residual connection
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x