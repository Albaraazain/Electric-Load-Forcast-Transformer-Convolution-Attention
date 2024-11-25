# transformer_conv_attention/models/transformer/layers/conv_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from ....utils.logger import get_logger

logger = get_logger(__name__)


def safe_softmax(attn_scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax with numerical stability"""
    attn_scores = attn_scores - attn_scores.max(dim=dim, keepdim=True)[0]
    exp_scores = torch.exp(attn_scores)
    return exp_scores / (exp_scores.sum(dim=dim, keepdim=True) + 1e-9)


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
        """Compute convolutional attention"""
        batch_size = query.size(0)
        """
        Compute convolutional attention
        """
        print(f"\n[DEBUG] ConvAttn Input Stats:")
        print(f"[DEBUG] - query: min={query.min():.4f}, max={query.max():.4f}, mean={query.mean():.4f}")
        print(f"[DEBUG] - key: min={key.min():.4f}, max={key.max():.4f}, mean={key.mean():.4f}")
        print(f"[DEBUG] - value: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")

        batch_size = query.size(0)

        # Apply convolutions
        # Transpose for conv1d: [batch_size, d_model, seq_len]
        q = self.conv_q(query.transpose(1, 2)).transpose(1, 2)
        k = self.conv_k(key.transpose(1, 2)).transpose(1, 2)
        v = self.conv_v(value.transpose(1, 2)).transpose(1, 2)

        print(f"[DEBUG] After Convolutions:")
        print(f"[DEBUG] - q: min={q.min():.4f}, max={q.max():.4f}, mean={q.mean():.4f}")
        print(f"[DEBUG] - k: min={k.min():.4f}, max={k.max():.4f}, mean={k.mean():.4f}")
        print(f"[DEBUG] - v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

        # Reshape for multi-head attention
        q = q.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)
        k = k.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)
        v = v.contiguous().view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        print(f"[DEBUG] Attention Scores Before Mask:")
        print(
            f"[DEBUG] - scores: min={attn_scores.min():.4f}, max={attn_scores.max():.4f}, mean={attn_scores.mean():.4f}")

        # Apply mask if provided
        if mask is not None:
            print(f"[DEBUG] Applying mask: {mask.shape}")
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            print(
                f"[DEBUG] After mask: min={attn_scores.min():.4f}, max={attn_scores.max():.4f}, mean={attn_scores.mean():.4f}")

        # Apply softmax and dropout
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask if provided with numerical stability
        if mask is not None:
            # Expand mask for batch size and heads
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
            mask = mask.expand(batch_size, self.n_heads, -1, -1)

            # Use a large negative number instead of -inf
            masked_scores = attn_scores.masked_fill(mask, -1e9)
        else:
            masked_scores = attn_scores

        # Numerically stable softmax
        scores_max, _ = masked_scores.max(dim=-1, keepdim=True)
        exp_scores = torch.exp(masked_scores - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Apply dropout
        attn_weights = self.dropout_layer(attn_weights)
        # Compute output
        output = torch.matmul(attn_weights, v)
        print(f"[DEBUG] After attention computation:")
        print(f"[DEBUG] - output: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        print(f"[DEBUG] Final output:")
        print(f"[DEBUG] - output: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

        if need_weights:
            return output, attn_weights
        else:
            return output, None
