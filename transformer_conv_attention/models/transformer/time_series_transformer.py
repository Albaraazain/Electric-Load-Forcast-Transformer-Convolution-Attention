# transformer_conv_attention/models/transformer/time_series_transformer.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from ..interfaces.time_series_model import TimeSeriesModel
from .conv_transformer import ConvolutionalTransformer
from ...utils.logger import get_logger

logger = get_logger(__name__)

# transformer_conv_attention/models/transformer/time_series_transformer.py
class TimeSeriesTransformerModel(TimeSeriesModel, nn.Module):
    """Time series specific transformer model"""

    def __init__(self, d_model, n_heads, n_encoder_layers, n_decoder_layers,
                 d_ff, kernel_size, max_seq_length, input_size, output_size, dropout=0.1):
        """Initialize time series transformer"""
        super().__init__()  # Initialize both parent classes

        print(f"[DEBUG] Initializing TimeSeriesTransformerModel:")
        print(f"[DEBUG] - input_size: {input_size}")
        print(f"[DEBUG] - d_model: {d_model}")

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
            src_mask: Optional attention mask

        Returns:
            - Predictions [batch_size, target_len, output_size]
            - Attention weights dictionary
        """
        print(f"[DEBUG] Forward pass input stats:")
        print(f"[DEBUG] - x: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Project input to model dimension
        enc_input = self.input_projection(x)
        print(f"[DEBUG] After input projection:")
        print(f"[DEBUG] - enc_input: min={enc_input.min():.4f}, max={enc_input.max():.4f}, mean={enc_input.mean():.4f}")

        # Initialize decoder input with zeros
        batch_size = x.size(0)
        device = x.device
        dec_input = torch.zeros(batch_size, target_len, self.input_size, device=device)
        dec_input = self.input_projection(dec_input)

        # Create decoder causal mask
        tgt_mask = self._generate_square_subsequent_mask(target_len).to(x.device)
        tgt_mask = ~tgt_mask  # Invert the mask for masked_fill operation


    # Transformer forward pass
        transformer_output = self.transformer(
            src=enc_input,
            tgt=dec_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        print(f"[DEBUG] After transformer:")
        print(f"[DEBUG] - transformer_output: min={transformer_output.min():.4f}, max={transformer_output.max():.4f}, mean={transformer_output.mean():.4f}")

        # Project output
        predictions = self.output_projection(transformer_output)
        print(f"[DEBUG] Final predictions:")
        print(f"[DEBUG] - predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

        # Get attention weights
        attention_weights = self.transformer.get_attention_weights()

        return predictions, attention_weights

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get attention maps from last forward pass"""
        return self.transformer.get_attention_weights()

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        # Create a mask where later positions are masked out
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()  # True values will be masked