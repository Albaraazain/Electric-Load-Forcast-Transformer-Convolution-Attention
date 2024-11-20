from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    """Configuration for transformer model"""
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    kernel_size: int
    max_seq_length: int
    learning_rate: float
    batch_size: int

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)