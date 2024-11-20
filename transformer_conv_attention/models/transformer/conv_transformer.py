# transformer_conv_attention/models/transformer/conv_transformer.py
import torch
import torch.nn as nn
from typing import Optional, Dict
from ..interfaces.transformer_interface import TransformerBase
from .layers.position_encoding import PositionalEncoding
from .layers.encoder_layer import EncoderLayer
from .layers.decoder_layer import DecoderLayer
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ConvolutionalTransformer(TransformerBase, nn.Module):  # Add nn.Module inheritance
    """Transformer model with convolutional attention"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            d_ff: int,
            kernel_size: int,
            max_seq_length: int,
            dropout: float = 0.1
    ):
        """Initialize transformer model"""
        super().__init__()  # Initialize both parent classes

        # Save config for attention weights
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        # Encoder
        encoder_layer = EncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = encoder_norm

        # Decoder
        decoder_layer = DecoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = decoder_norm

        logger.info(
            f"Initialized transformer with d_model={d_model}, "
            f"n_heads={n_heads}, n_encoder_layers={n_encoder_layers}, "
            f"n_decoder_layers={n_decoder_layers}"
        )

    def encode(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input sequence"""
        x = self.pos_encoding(src)

        for layer in self.encoder:
            x = layer(x, src_mask)

        return self.encoder_norm(x)

    def decode(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode output sequence"""
        x = self.pos_encoding(tgt)

        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, memory_mask)

        return self.decoder_norm(x)

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
            src: Source sequence [batch_size, src_len, d_model]
            tgt: Target sequence [batch_size, tgt_len, d_model]
            src_mask: Source mask
            tgt_mask: Target mask
            memory_mask: Memory mask

        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        return output

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights from last forward pass"""
        weights = {
            'encoder_self_attention': [layer.attn_weights for layer in self.encoder],
            'decoder_self_attention': [layer.self_attn_weights for layer in self.decoder],
            'decoder_cross_attention': [layer.cross_attn_weights for layer in self.decoder]
        }
        return weights