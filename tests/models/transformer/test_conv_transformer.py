# tests/models/transformer/test_conv_transformer.py
import pytest
import torch

from transformer_conv_attention.models.builders.conv_transformer_builder import ConvTransformerBuilder
from transformer_conv_attention.models.transformer.conv_transformer import ConvolutionalTransformer
from transformer_conv_attention.models.factory.model_factory import ModelFactory
from transformer_conv_attention.config.model_config import TransformerConfig

@pytest.fixture
def model_config():
    return {
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'kernel_size': 3,
        'max_seq_length': 1000,
        'dropout': 0.1
    }

def test_transformer_initialization(model_config):
    model = ConvolutionalTransformer(**model_config)
    assert isinstance(model, ConvolutionalTransformer)
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')
    assert hasattr(model, 'pos_encoding')

def test_transformer_forward(model_config):
    batch_size = 32
    src_len = 100
    tgt_len = 50

    model = ConvolutionalTransformer(**model_config)
    src = torch.randn(batch_size, src_len, model_config['d_model'])
    tgt = torch.randn(batch_size, tgt_len, model_config['d_model'])

    output = model(src, tgt)
    assert output.shape == (batch_size, tgt_len, model_config['d_model'])

def test_transformer_masks(model_config):
    batch_size = 32
    src_len = 100
    tgt_len = 50

    model = ConvolutionalTransformer(**model_config)
    src = torch.randn(batch_size, src_len, model_config['d_model'])
    tgt = torch.randn(batch_size, tgt_len, model_config['d_model'])

    # Create masks
    src_mask = torch.triu(torch.ones(src_len, src_len), diagonal=1).bool()
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    memory_mask = torch.ones(tgt_len, src_len).bool()

    output = model(
        src,
        tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask
    )
    assert output.shape == (batch_size, tgt_len, model_config['d_model'])

def test_attention_weights(model_config):
    batch_size = 32
    src_len = 100
    tgt_len = 50

    model = ConvolutionalTransformer(**model_config)
    src = torch.randn(batch_size, src_len, model_config['d_model'])
    tgt = torch.randn(batch_size, tgt_len, model_config['d_model'])

    _ = model(src, tgt)
    weights = model.get_attention_weights()

    assert 'encoder_self_attention' in weights
    assert 'decoder_self_attention' in weights
    assert 'decoder_cross_attention' in weights

    assert len(weights['encoder_self_attention']) == model_config['n_encoder_layers']
    assert len(weights['decoder_self_attention']) == model_config['n_decoder_layers']
    assert len(weights['decoder_cross_attention']) == model_config['n_decoder_layers']

def test_model_factory():
    config = TransformerConfig(
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        kernel_size=3,
        max_seq_length=1000,
        dropout=0.1
    )

    ModelFactory.register_builder('conv_transformer', ConvTransformerBuilder)
    model = ModelFactory.create_model('conv_transformer', config)

    assert isinstance(model, ConvolutionalTransformer)

@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("seq_length", [10, 100])
def test_different_batch_sizes_and_lengths(model_config, batch_size, seq_length):
    model = ConvolutionalTransformer(**model_config)
    src = torch.randn(batch_size, seq_length, model_config['d_model'])
    tgt = torch.randn(batch_size, seq_length, model_config['d_model'])

    output = model(src, tgt)
    assert output.shape == (batch_size, seq_length, model_config['d_model'])

def test_gpu_support(model_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device('cuda')
    model = ConvolutionalTransformer(**model_config).to(device)

    batch_size = 32
    seq_length = 100
    src = torch.randn(batch_size, seq_length, model_config['d_model']).to(device)
    tgt = torch.randn(batch_size, seq_length, model_config['d_model']).to(device)

    output = model(src, tgt)
    # Change the assertion to check if the device type matches
    assert output.device.type == device.type