# tests/main_pipeline/test_stages.py
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from transformer_conv_attention.main_pipeline.stages import (
    DataLoadingStage,
    ModelSetupStage,
    TrainingStage,
    EvaluationStage
)
from transformer_conv_attention.main_pipeline.interfaces.pipeline_stage import PipelineContext
from transformer_conv_attention.models.registery import register_builders


@pytest.fixture
def sample_config():
    """Create a complete config for testing"""
    model_config = {
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'kernel_size': 3,
        'max_seq_length': 1000,
        'dropout': 0.1,
        'input_size': 10,
        'output_size': 1
    }

    from transformer_conv_attention.main_pipeline.pipeline_config import PipelineConfig
    pipeline_config = PipelineConfig(
        # Data config
        data_path="test_data.csv",
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2"],
        batch_size=32,
        num_workers=4,

        # Model config
        model_type="transformer",
        model_config=model_config,

        # Training config
        max_epochs=100,
        learning_rate=0.001,
        early_stopping_patience=10,

        # Time Series specific config
        window_size=48,  # Reduced for testing
        prediction_horizon=24,

        # Evaluation config
        eval_metrics=["mape", "rmse"]
    )

    return pipeline_config

@pytest.fixture
def pipeline_context(sample_config):
    context = PipelineContext()
    context.set_data('config', sample_config)
    return context

def test_data_loading_stage(pipeline_context, tmp_path):
    # Create test data file with enough samples
    data_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),  # More data
        'target': np.random.randn(1000),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    })
    df.to_csv(data_path, index=False)

    pipeline_context.get_data('config').data_path = str(data_path)

    stage = DataLoadingStage()
    result_context = stage.execute(pipeline_context)

    assert result_context.get_data('train_loader') is not None
    assert result_context.get_data('val_loader') is not None
    assert result_context.get_data('test_loader') is not None
    assert result_context.get_data('data_processor') is not None

class MockTimeSeriesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(10, 64)
        self.decoder = torch.nn.Linear(64, 1)

    def forward(self, x, target_len):
        batch_size = x.size(0)
        # Generate features that require grad
        features = self.encoder(x)

        # Generate predictions for each target step
        predictions = []
        for _ in range(target_len):
            output = self.decoder(features)
            predictions.append(output)

        predictions = torch.stack(predictions, dim=1)

        # Generate fake attention weights
        attention_weights = {
            'encoder_attention': torch.randn(batch_size, 8, 48, 48),
            'decoder_attention': torch.randn(batch_size, 8, target_len, 48)
        }

        return predictions, attention_weights
def create_mock_model():
    """Create a mock model that handles target_len parameter"""
    return MockTimeSeriesModel()

def create_mock_dataloader():
    """Create mock dataloader with correct dimensions"""
    input_size = 10
    target_size = 1
    num_samples = 10

    # Generate data that requires grad
    dataset = [
        (torch.randn(input_size).requires_grad_(True),
         torch.randn(24, target_size).requires_grad_(True))
        for _ in range(num_samples)
    ]

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda batch: (
            torch.stack([x[0] for x in batch]).requires_grad_(True),
            torch.stack([x[1] for x in batch]).requires_grad_(True)
        )
    )

def create_mock_processor():
    """Create mock data processor"""
    class MockProcessor:
        def transform(self, x):
            return x
        def inverse_transform(self, x):
            return x
    return MockProcessor()

def test_model_setup_stage(pipeline_context):
    # Register builders before testing
    register_builders()

    stage = ModelSetupStage()
    result_context = stage.execute(pipeline_context)

    model = result_context.get_data('model')
    device = result_context.get_data('device')

    assert model is not None
    assert device.type == 'cuda'
    assert next(model.parameters()).device.type == device.type

def test_training_stage(pipeline_context):
    # Setup mock data and model
    model = create_mock_model()
    train_loader = create_mock_dataloader()
    val_loader = create_mock_dataloader()

    pipeline_context.set_data('model', model)
    pipeline_context.set_data('train_loader', train_loader)
    pipeline_context.set_data('val_loader', val_loader)
    pipeline_context.set_data('device', torch.device('cpu'))

    stage = TrainingStage()
    result_context = stage.execute(pipeline_context)

    assert result_context.get_data('training_history') is not None
    assert result_context.get_data('trained_model') is not None

def test_evaluation_stage(pipeline_context):
    # Setup mock trained model and test data
    model = create_mock_model()
    test_loader = create_mock_dataloader()
    processor = create_mock_processor()

    pipeline_context.set_data('trained_model', model)
    pipeline_context.set_data('test_loader', test_loader)
    pipeline_context.set_data('data_processor', processor)
    pipeline_context.set_data('device', torch.device('cpu'))

    stage = EvaluationStage()
    result_context = stage.execute(pipeline_context)

    assert result_context.get_data('evaluation_metrics') is not None
    assert result_context.get_data('attention_weights') is not None