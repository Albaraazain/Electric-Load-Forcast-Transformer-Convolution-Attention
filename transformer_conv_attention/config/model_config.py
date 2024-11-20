# transformer_conv_attention/config/model_config.py
from dataclasses import dataclass
from typing import Optional
from .base_config import BaseConfig
from ..utils.exceptions import ConfigurationError

@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for transformer model architecture"""
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    kernel_size: int
    max_seq_length: int

    def validate(self) -> None:
        """Validate configuration values"""
        if self.d_model <= 0:
            raise ConfigurationError("d_model must be positive")
        if self.d_model % self.n_heads != 0:
            raise ConfigurationError("d_model must be divisible by n_heads")
        if self.kernel_size % 2 != 1:
            raise ConfigurationError("kernel_size must be odd")

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training"""
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: Optional[int] = None
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gradient_clip_val: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration values"""
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ConfigurationError("max_epochs must be positive")