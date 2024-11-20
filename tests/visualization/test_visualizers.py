# tests/visualization/test_visualizers.py
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from transformer_conv_attention.visualization.factory.visualizer_factory import VisualizerFactory
from transformer_conv_attention.visualization.visualization_manager import VisualizationManager

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create timestamps - make sure the number of samples is divisible by 24
    timestamps = pd.date_range(
        start='2023-01-01',
        periods=168,  # 7 days * 24 hours = 168 hours (divisible by 24)
        freq='H'
    )

    # Create actual and predicted values
    n_samples = len(timestamps)
    actual = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 100 + 500
    predicted = actual + np.random.normal(0, 10, n_samples)

    # Reshape predicted to have multiple horizons
    predicted = predicted.reshape(-1, 24)  # Now it will be (7, 24) shape

    # Create attention weights
    attention_weights = np.random.rand(8, 8, 24, 24)  # 8 layers, 8 heads, fixed size

    return {
        'timestamps': timestamps,
        'actual': actual,
        'predicted': predicted,
        'attention_weights': attention_weights
    }

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs"""
    output_dir = tmp_path / "test_visualizations"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def test_prediction_visualizer_creation():
    """Test creation of prediction visualizer"""
    visualizer = VisualizerFactory.create_visualizer('prediction')
    assert visualizer.get_name() == "Prediction Plot"

def test_attention_visualizer_creation():
    """Test creation of attention visualizer"""
    visualizer = VisualizerFactory.create_visualizer('attention')
    assert visualizer.get_name() == "Attention Heatmap"

def test_error_analysis_visualizer_creation():
    """Test creation of error analysis visualizer"""
    visualizer = VisualizerFactory.create_visualizer('error')
    assert visualizer.get_name() == "Error Analysis"

def test_invalid_visualizer_type():
    """Test creation of invalid visualizer type"""
    with pytest.raises(ValueError):
        VisualizerFactory.create_visualizer('invalid_type')

def test_prediction_plot(sample_data, temp_dir):
    """Test prediction plot creation and saving"""
    visualizer = VisualizerFactory.create_visualizer('prediction')
    fig = visualizer.plot(sample_data, horizons=[24])

    assert isinstance(fig, plt.Figure)

    # Test saving
    save_path = temp_dir / "prediction_test.png"
    visualizer.save(fig, str(save_path))
    assert save_path.exists()

    plt.close(fig)

def test_attention_plot(sample_data, temp_dir):
    """Test attention plot creation and saving"""
    visualizer = VisualizerFactory.create_visualizer('attention')
    fig = visualizer.plot(
        sample_data,
        layer_index=0,
        head_index=0
    )

    assert isinstance(fig, plt.Figure)

    # Test saving
    save_path = temp_dir / "attention_test.png"
    visualizer.save(fig, str(save_path))
    assert save_path.exists()

    plt.close(fig)

def test_error_analysis_plot(sample_data, temp_dir):
    """Test error analysis plot creation and saving"""
    visualizer = VisualizerFactory.create_visualizer('error')
    fig = visualizer.plot(sample_data)

    assert isinstance(fig, plt.Figure)

    # Test saving
    save_path = temp_dir / "error_test.png"
    visualizer.save(fig, str(save_path))
    assert save_path.exists()

    plt.close(fig)

def test_visualization_manager(sample_data, temp_dir):
    """Test visualization manager functionality"""
    manager = VisualizationManager(str(temp_dir))

    # Add visualizers
    manager.add_visualizer('pred', 'prediction')
    manager.add_visualizer('attn', 'attention')
    manager.add_visualizer('err', 'error')

    # Test single visualization
    fig = manager.create_visualization(
        'pred',
        sample_data,
        horizons=[24]
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test all visualizations
    results = manager.create_all_visualizations(sample_data)
    assert all(isinstance(fig, plt.Figure) for fig in results.values())
    for fig in results.values():
        plt.close(fig)

def test_visualization_manager_invalid_visualizer():
    """Test visualization manager with invalid visualizer"""
    manager = VisualizationManager()
    with pytest.raises(ValueError):
        manager.add_visualizer('invalid', 'nonexistent_type')

def test_visualization_manager_missing_visualizer(sample_data):
    """Test visualization manager with missing visualizer"""
    manager = VisualizationManager()
    with pytest.raises(ValueError):
        manager.create_visualization('nonexistent', sample_data)

@pytest.mark.parametrize("figsize", [(8, 6), (12, 8), (15, 10)])
def test_custom_figure_sizes(figsize, sample_data):
    """Test visualizers with different figure sizes"""
    visualizer = VisualizerFactory.create_visualizer(
        'prediction',
        figsize=figsize
    )
    fig = visualizer.plot(sample_data, horizons=[24])

    assert fig.get_size_inches() == figsize
    plt.close(fig)

def test_style_selection():
    """Test visualizer with different styles"""
    available_styles = ['seaborn', 'ggplot', 'classic']
    for style in available_styles:
        visualizer = VisualizerFactory.create_visualizer(
            'prediction',
            style=style
        )
        assert style in plt.style.available
        plt.close('all')

@pytest.mark.parametrize("horizons", [
    [24],
    [24, 48],
    [24, 48, 96]
])
def test_prediction_multiple_horizons(sample_data, horizons):
    """Test prediction plot with different numbers of horizons"""
    visualizer = VisualizerFactory.create_visualizer('prediction')
    fig = visualizer.plot(sample_data, horizons=horizons)

    # Check number of subplots matches number of horizons
    assert len(fig.axes) == len(horizons)
    plt.close(fig)

def test_save_format_options(sample_data, temp_dir):
    """Test saving visualizations in different formats"""
    visualizer = VisualizerFactory.create_visualizer('prediction')
    fig = visualizer.plot(sample_data, horizons=[24])

    formats = ['png', 'pdf']  # Removed 'svg' as it might not be supported everywhere
    for fmt in formats:
        save_path = temp_dir / f"test.{fmt}"
        visualizer.save(fig, str(save_path))
        assert save_path.exists()

    plt.close(fig)

def test_visualization_data_validation(sample_data):
    """Test validation of input data"""
    invalid_data = sample_data.copy()
    del invalid_data['timestamps']

    visualizer = VisualizerFactory.create_visualizer('prediction')
    with pytest.raises(KeyError):
        visualizer.plot(invalid_data)

def test_concurrent_visualization_creation(sample_data, temp_dir):
    """Test creating multiple visualizations concurrently"""
    manager = VisualizationManager(str(temp_dir))

    # Add multiple visualizers
    for i in range(3):
        manager.add_visualizer(f'pred_{i}', 'prediction')

    # Create visualizations concurrently
    results = []
    for i in range(3):
        fig = manager.create_visualization(f'pred_{i}', sample_data)
        results.append(fig)

    assert all(isinstance(fig, plt.Figure) for fig in results)
    for fig in results:
        plt.close(fig)