# tests/visualization/test_visualization_integration.py
import pandas as pd
import pytest
import numpy as np
import torch
from matplotlib import pyplot as plt

from transformer_conv_attention.visualization.visualization_manager import VisualizationManager
from transformer_conv_attention.evaluation.evaluator import TimeSeriesEvaluator
from transformer_conv_attention.models.transformer.time_series_transformer import TimeSeriesTransformerModel

@pytest.fixture
def model_and_data():
    """Create sample model and data for integration testing"""
    # Create a small model
    model = TimeSeriesTransformerModel(
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=64,
        kernel_size=3,
        max_seq_length=100,
        input_size=1,
        output_size=1
    )

    # Create sample data
    batch_size = 4
    seq_length = 48
    x = torch.randn(batch_size, seq_length, 1)
    y = torch.randn(batch_size, 24, 1)  # 24-hour prediction

    return model, x, y

def test_visualization_with_model_output(model_and_data, temp_dir):
    """Test visualization pipeline with actual model output"""
    model, x, y = model_and_data

    # Get model predictions
    with torch.no_grad():
        predictions, attention_weights = model(x, target_len=24)

    # Prepare data for visualization
    data = {
        'timestamps': pd.date_range('2023-01-01', periods=len(x), freq='H'),
        'actual': y.numpy(),
        'predicted': predictions.numpy(),
        'attention_weights': {k: v.numpy() for k, v in attention_weights.items()}
    }

    # Create visualizations
    viz_manager = VisualizationManager(str(temp_dir))
    viz_manager.add_visualizer('predictions', 'prediction')
    viz_manager.add_visualizer('attention', 'attention')
    viz_manager.add_visualizer('error', 'error')

    results = viz_manager.create_all_visualizations(data)
    assert all(isinstance(fig, plt.Figure) for fig in results.values())

    # Clean up
    for fig in results.values():
        plt.close(fig)

def test_visualization_with_evaluator(model_and_data, temp_dir):
    """Test visualization integration with evaluator"""
    model, x, y = model_and_data

    # Create evaluator
    evaluator = TimeSeriesEvaluator(model, device='cpu')

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Get evaluation results
    metrics, attention_weights = evaluator.evaluate(dataloader, prediction_length=24)

    # Create visualizations
    viz_manager = VisualizationManager(str(temp_dir))
    viz_manager.add_visualizer('error', 'error')

    # Prepare data
    data = {
        'timestamps': pd.date_range('2023-01-01', periods=len(x), freq='H'),
        'actual': y.numpy(),
        'predicted': predictions.numpy(),
        'metrics': metrics,
        'attention_weights': attention_weights
    }

    results = viz_manager.create_all_visualizations(data)
    assert all(isinstance(fig, plt.Figure) for fig in results.values())

    # Clean up
    for fig in results.values():
        plt.close(fig)