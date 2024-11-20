# tests/config/test_config.py
import pytest
from pathlib import Path
from transformer_conv_attention.config import TransformerConfig, TrainingConfig
from transformer_conv_attention.utils.exceptions import ConfigurationError

def test_transformer_config_validation():
    # Valid configuration
    valid_config = TransformerConfig(
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        kernel_size=3,
        max_seq_length=1000
    )
    valid_config.validate()  # Should not raise

    # Invalid configurations
    with pytest.raises(ConfigurationError):
        invalid_config = TransformerConfig(
            d_model=512,
            n_heads=7,  # Not divisible by d_model
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            kernel_size=3,
            max_seq_length=1000
        )
        invalid_config.validate()