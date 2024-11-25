from typing import Any
import torch.nn as nn
from .base_builder import ModelBuilder
from ..transformer.time_series_transformer import TimeSeriesTransformerModel
from ...utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesTransformerBuilder(ModelBuilder):
    """Builder for time series transformer model"""

    def build_model(self, config: Any) -> nn.Module:
        return TimeSeriesTransformerModel(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_ff=config.d_ff,
            kernel_size=config.kernel_size,
            max_seq_length=config.max_seq_length,
            input_size=config.input_size,
            output_size=config.output_size,
            dropout=config.dropout
        )