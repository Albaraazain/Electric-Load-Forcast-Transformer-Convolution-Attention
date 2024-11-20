# tests/models/transformer/layers/test_encoder_decoder_layers.py
import pytest
import torch
from transformer_conv_attention.models.transformer.layers.encoder_layer import EncoderLayer
from transformer_conv_attention.models.transformer.layers.decoder_layer import DecoderLayer

@pytest.fixture
def layer_config():
    return {
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'kernel_size': 3,
        'dropout': 0.1
    }

def test_encoder_layer(layer_config):
    batch_size = 32
    seq_length = 100

    layer = EncoderLayer(**layer_config)
    x = torch.randn(batch_size, seq_length, layer_config['d_model'])

    output = layer(x)

    assert output.shape == (batch_size, seq_length, layer_config['d_model'])
    assert hasattr(layer, 'attn_weights')

def test_decoder_layer(layer_config):
    batch_size = 32
    tgt_len = 50
    src_len = 100

    layer = DecoderLayer(**layer_config)
    x = torch.randn(batch_size, tgt_len, layer_config['d_model'])
    memory = torch.randn(batch_size, src_len, layer_config['d_model'])

    output = layer(x, memory)

    assert output.shape == (batch_size, tgt_len, layer_config['d_model'])
    assert hasattr(layer, 'self_attn_weights')
    assert hasattr(layer, 'cross_attn_weights')

def test_encoder_layer_mask(layer_config):
    batch_size = 32
    seq_length = 100

    layer = EncoderLayer(**layer_config)
    x = torch.randn(batch_size, seq_length, layer_config['d_model'])
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

    output = layer(x, mask=mask)
    assert output.shape == (batch_size, seq_length, layer_config['d_model'])

def test_decoder_layer_masks(layer_config):
    batch_size = 32
    tgt_len = 50
    src_len = 100

    layer = DecoderLayer(**layer_config)
    x = torch.randn(batch_size, tgt_len, layer_config['d_model'])
    memory = torch.randn(batch_size, src_len, layer_config['d_model'])

    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    memory_mask = torch.ones(tgt_len, src_len).bool()

    output = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
    assert output.shape == (batch_size, tgt_len, layer_config['d_model'])