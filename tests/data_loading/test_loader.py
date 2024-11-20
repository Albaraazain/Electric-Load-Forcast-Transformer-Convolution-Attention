# tests/data_loading/test_loader.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from transformer_conv_attention.data_loading.loaders.time_series_loader import TimeSeriesLoader
from transformer_conv_attention.utils import DataError


@pytest.fixture
def sample_data_file(tmp_path):
    """Create sample data file"""
    dates = pd.date_range(
        start='2023-01-01',
        end='2023-01-31',
        freq='H'
    )

    df = pd.DataFrame({
        'timestamp': dates,
        'target': np.sin(np.arange(len(dates)) * 2 * np.pi / 24),
        'feature1': np.random.normal(0, 1, len(dates)),
        'feature2': np.random.normal(0, 1, len(dates))
    })

    # Save to CSV
    file_path = tmp_path / "sample_data.csv"
    df.to_csv(file_path, index=False)

    return file_path

def test_loader_initialization(sample_data_file):
    loader = TimeSeriesLoader(
        data_path=str(sample_data_file),
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2'],
        window_size=24,
        prediction_horizon=12
    )

    # Check if data was split correctly
    assert len(loader.train_data) > 0
    assert len(loader.val_data) > 0
    assert len(loader.test_data) > 0

def test_get_dataloaders(sample_data_file):
    loader = TimeSeriesLoader(
        data_path=str(sample_data_file),
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2'],
        window_size=24,
        prediction_horizon=12,
        batch_size=32
    )

    train_loader, val_loader, test_loader = loader.get_dataloaders()

    # Check if dataloaders are created correctly
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check batch from train loader
    batch = next(iter(train_loader))
    encoder_input, decoder_input, target = batch

    assert len(batch) == 3
    assert encoder_input.shape[1] == 24  # window_size
    assert decoder_input.shape[1] == 12  # prediction_horizon
    assert target.shape[1] == 12  # prediction_horizon

def test_processor_save_load(sample_data_file, tmp_path):
    loader = TimeSeriesLoader(
        data_path=str(sample_data_file),
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2']
    )

    # Save processor
    processor_path = tmp_path / "processor.pkl"
    loader.save_processor(processor_path)

    # Create new loader and load processor
    new_loader = TimeSeriesLoader(
        data_path=str(sample_data_file),
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2']
    )
    new_loader.load_processor(processor_path)

    # Check if scalers match
    assert loader.processor.scalers == new_loader.processor.scalers

def test_invalid_file_path():
    with pytest.raises(DataError, match="Failed to load data"):
        TimeSeriesLoader(
            data_path="nonexistent.csv",
            target_column='target',
            timestamp_column='timestamp'
        )

def test_invalid_data_format(tmp_path):
    # Create invalid data file
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("invalid data")

    with pytest.raises(DataError, match="Unsupported file format"):
        TimeSeriesLoader(
            data_path=str(invalid_file),
            target_column='target',
            timestamp_column='timestamp'
        )