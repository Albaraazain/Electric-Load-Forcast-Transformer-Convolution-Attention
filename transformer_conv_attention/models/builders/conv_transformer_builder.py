# transformer_conv_attention/models/builders/conv_transformer_builder.py
import torch.nn as nn
from typing import Dict, Any
from ..interfaces.model_builder import ModelBuilder
from ..transformer.conv_transformer import ConvolutionalTransformer
from ..transformer.layers.conv_attention import ConvolutionalAttention
from ..transformer.layers.position_encoding import PositionalEncoding
from ..transformer.layers.encoder_layer import EncoderLayer
from ..transformer.layers.decoder_layer import DecoderLayer
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ConvTransformerBuilder(ModelBuilder):
    """Builder for convolutional transformer model"""

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
        """Build transformer encoder"""
        encoder_layer = self.build_encoder_layer(config)
        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_encoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )
        return encoder

    def build_decoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build transformer decoder"""
        decoder_layer = self.build_decoder_layer(config)
        decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['n_decoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )
        return decoder

    def build_model(self, config: Dict[str, Any]) -> ConvolutionalTransformer:
        """Build complete transformer model"""
        return ConvolutionalTransformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            d_ff=config['d_ff'],
            kernel_size=config['kernel_size'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout']
        )