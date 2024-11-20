# transformer_conv_attention/models/transformer/time_series_transformer.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from ..interfaces.time_series_model import TimeSeriesModel
from .conv_transformer import ConvolutionalTransformer
from ...utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesTransformerModel(TimeSeriesModel, nn.Module):
    """Time series specific transformer model"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            d_ff: int,
            kernel_size: int,
            max_seq_length: int,
            input_size: int,
            output_size: int,
            dropout: float = 0.1
    ):
        """Initialize time series transformer"""
        super().__init__()  # Initialize both parent classes

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model

        # Input and output projections
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, output_size)

        # Core transformer
        self.transformer = ConvolutionalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            d_ff=d_ff,
            kernel_size=kernel_size,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        logger.info(
            f"Initialized time series transformer with input_size={input_size}, "
            f"output_size={output_size}, d_model={d_model}"
        )

    def forward(
            self,
            x: torch.Tensor,
            target_len: int,
            src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_length, input_size]
            target_len: Number of time steps to predict
            src_mask: Optional mask for encoder

        Returns:
            - Predictions [batch_size, target_len, output_size]
            - Attention weights dictionary
        """
        batch_size = x.size(0)
        device = x.device

        # Project input to model dimension
        enc_input = self.input_projection(x)  # [batch_size, seq_length, d_model]

        # Initialize decoder input with zeros
        dec_input = torch.zeros(batch_size, target_len, self.input_size, device=device)
        dec_input = self.input_projection(dec_input)  # [batch_size, target_len, d_model]

        # Create decoder causal mask
        tgt_mask = self._generate_square_subsequent_mask(target_len).to(device)

        # Transformer forward pass
        transformer_output = self.transformer(
            src=enc_input,  # [batch_size, seq_length, d_model]
            tgt=dec_input,  # [batch_size, target_len, d_model]
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        # Project output
        predictions = self.output_projection(transformer_output)  # [batch_size, target_len, output_size]

        # Get attention weights
        attention_weights = self.transformer.get_attention_weights()

        return predictions, attention_weights

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get attention maps from last forward pass"""
        return self.transformer.get_attention_weights()
