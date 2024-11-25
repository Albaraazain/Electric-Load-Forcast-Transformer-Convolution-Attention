# tests/models/transformer/test_time_series_transformer.py
import pytest
import torch

from transformer_conv_attention.models.builders.time_series_builder import TimeSeriesTransformerBuilder
from transformer_conv_attention.models.transformer.time_series_transformer import TimeSeriesTransformerModel
from transformer_conv_attention.config.time_series_config import TimeSeriesConfig
from transformer_conv_attention.models import ModelFactory

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
        'input_size': 10,
        'output_size': 1,
        'dropout': 0.1
    }

def test_time_series_model_initialization(model_config):
    model = TimeSeriesTransformerModel(**model_config)
    assert isinstance(model, TimeSeriesTransformerModel)
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'input_projection')
    assert hasattr(model, 'output_projection')

def test_time_series_forward(model_config):
    batch_size = 32
    seq_length = 100
    target_len = 24

    model = TimeSeriesTransformerModel(**model_config)
    x = torch.randn(batch_size, seq_length, model_config['input_size'])

    predictions, attention_weights = model(x, target_len)

    assert predictions.shape == (batch_size, target_len, model_config['output_size'])
    assert 'encoder_self_attention' in attention_weights
    assert 'decoder_self_attention' in attention_weights
    assert 'decoder_cross_attention' in attention_weights

def test_time_series_config():
    config = TimeSeriesConfig(
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        kernel_size=3,
        max_seq_length=1000,
        dropout=0.1,
        input_size=10,
        output_size=1,
        sequence_length=168,
        prediction_length=24,
        input_features=['feature1', 'feature2'],
        target_features=['target'],
        scaling_method='standard'
    )

    assert config.input_size == 10
    assert config.output_size == 1
    assert len(config.input_features) == 2
    assert len(config.target_features) == 1

def test_time_series_config_validation():
    with pytest.raises(ValueError):
        TimeSeriesConfig(
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
            kernel_size=3,
            max_seq_length=1000,
            dropout=0.1,
            input_size=-1,  # Invalid
            output_size=1,
            sequence_length=168,
            prediction_length=24,
            input_features=['feature1'],
            target_features=['target'],
            scaling_method='standard'
        )

@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("seq_length", [48, 168, 336])
@pytest.mark.parametrize("target_len", [24, 48, 96])
def test_different_sequence_lengths(model_config, batch_size, seq_length, target_len):
    model = TimeSeriesTransformerModel(**model_config)
    x = torch.randn(batch_size, seq_length, model_config['input_size'])

    predictions, _ = model(x, target_len)
    assert predictions.shape == (batch_size, target_len, model_config['output_size'])

def test_model_factory_with_time_series():
    config = TimeSeriesConfig(
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        kernel_size=3,
        max_seq_length=1000,
        dropout=0.1,
        input_size=10,
        output_size=1,
        sequence_length=168,
        prediction_length=24,
        input_features=['feature1', 'feature2'],
        target_features=['target'],
        scaling_method='standard'
    )

    ModelFactory.register_builder('time_series_transformer', TimeSeriesTransformerBuilder)
    model = ModelFactory.create_model('time_series_transformer', config)

    assert isinstance(model, TimeSeriesTransformerModel)

def test_gpu_support(model_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device('cuda')
    model = TimeSeriesTransformerModel(**model_config).to(device)

    batch_size = 32
    seq_length = 100
    target_len = 24
    x = torch.randn(batch_size, seq_length, model_config['input_size']).to(device)

    predictions, _ = model(x, target_len)
    assert predictions.device.type == device.type