# transformer_conv_attention/models/transformer/layers/conv_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from ....utils.logger import get_logger

logger = get_logger(__name__)

class ConvolutionalAttention(nn.Module):
    """Multi-head attention with convolutional feature extraction"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            kernel_size: int = 3,
            dropout: float = 0.1,
            bias: bool = True
    ):
        """
        Initialize convolutional attention

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            kernel_size: Kernel size for convolution
            dropout: Dropout rate
            bias: Whether to use bias in linear transformations
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.scale = math.sqrt(self.head_dim)

        # Convolutional layers for Q, K, V
        padding = (kernel_size - 1) // 2
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, bias=bias)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, bias=bias)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"Initialized convolutional attention with d_model={d_model}, "
            f"n_heads={n_heads}, kernel_size={kernel_size}"
        )

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute convolutional attention

        Args:
            query: Query tensor [batch_size, tgt_len, d_model]
            key: Key tensor [batch_size, src_len, d_model]
            value: Value tensor [batch_size, src_len, d_model]
            mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            - Output tensor [batch_size, tgt_len, d_model]
            - Attention weights (optional) [batch_size, n_heads, tgt_len, src_len]
        """
        batch_size = query.size(0)

        # Apply convolutions
        # Transpose for conv1d: [batch_size, d_model, seq_len]
        q = self.conv_q(query.transpose(1, 2)).transpose(1, 2)
        k = self.conv_k(key.transpose(1, 2)).transpose(1, 2)
        v = self.conv_v(value.transpose(1, 2)).transpose(1, 2)

        # Reshape for multi-head attention
        # [batch_size, seq_len, n_heads, head_dim]
        q = q.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)
        k = k.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)
        v = v.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention computation
        # [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        else:
            return output, None