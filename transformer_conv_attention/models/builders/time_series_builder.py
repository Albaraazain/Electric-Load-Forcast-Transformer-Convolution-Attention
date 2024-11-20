# transformer_conv_attention/models/builders/time_series_builder.py
from typing import Dict, Any

from torch import nn

from ..factory.model_factory import ModelFactory
from ..interfaces.model_builder import ModelBuilder
from ..transformer.layers.conv_attention import ConvolutionalAttention
from ..transformer.layers.decoder_layer import DecoderLayer
from ..transformer.layers.encoder_layer import EncoderLayer
from ..transformer.time_series_transformer import TimeSeriesTransformerModel
from ...utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesTransformerBuilder(ModelBuilder):
    """Builder for time series transformer model"""

    def build_attention(self, config: Dict[str, Any]) -> nn.Module:
        return ConvolutionalAttention(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )

    def build_encoder_layer(self, config: Dict[str, Any]) -> nn.Module:
        return EncoderLayer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )

    def build_decoder_layer(self, config: Dict[str, Any]) -> nn.Module:
        return DecoderLayer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )

    def build_encoder(self, config: Dict[str, Any]) -> nn.Module:
        encoder_layer = self.build_encoder_layer(config)

        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_encoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )

    def build_decoder(self, config: Dict[str, Any]) -> nn.Module:
        decoder_layer = self.build_decoder_layer(config)
        return nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['n_decoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )

    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        return TimeSeriesTransformerModel(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            d_ff=config['d_ff'],
            kernel_size=config['kernel_size'],
            max_seq_length=config['max_seq_length'],
            input_size=config['input_size'],
            output_size=config['output_size'],
            dropout=config['dropout']
        )




# Register the builder
ModelFactory.register_builder('time_series_transformer', TimeSeriesTransformerBuilder)
