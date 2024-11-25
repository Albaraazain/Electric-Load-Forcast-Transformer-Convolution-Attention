# tests/data_loading/test_dataset.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformer_conv_attention.data_loading.datasets.time_series_dataset import TransformerTimeSeriesDataset
from transformer_conv_attention.utils import DataError


@pytest.fixture
def sample_data():
    """Create sample time series data"""
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

    return df

def test_dataset_initialization(sample_data):
    dataset = TransformerTimeSeriesDataset(
        data=sample_data,
        window_size=24,
        prediction_horizon=12,
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2']
    )

    assert len(dataset) > 0
    assert dataset.get_feature_dim() == 8  # 2 features + 6 time features

def test_dataset_getitem(sample_data):
    dataset = TransformerTimeSeriesDataset(
        data=sample_data,
        window_size=24,
        prediction_horizon=12,
        target_column='target',
        timestamp_column='timestamp',
        feature_columns=['feature1', 'feature2']
    )

    encoder_input, decoder_input, target = dataset[0]

    assert encoder_input.shape == (24, 8)  # window_size x features
    assert decoder_input.shape == (12, 8)  # prediction_horizon x features
    assert target.shape == (12,)  # prediction_horizon

def test_invalid_window_size(sample_data):
    with pytest.raises(ValueError, match="window_size must be positive"):
        TransformerTimeSeriesDataset(
            data=sample_data,
            window_size=0,
            prediction_horizon=12,
            target_column='target',
            timestamp_column='timestamp'
        )

def test_invalid_column(sample_data):
    with pytest.raises(DataError, match="Missing required columns"):
        TransformerTimeSeriesDataset(
            data=sample_data,
            window_size=24,
            prediction_horizon=12,
            target_column='nonexistent',
            timestamp_column='timestamp'
        )