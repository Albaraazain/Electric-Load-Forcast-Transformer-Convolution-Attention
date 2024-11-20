# tests/data_loading/test_processor.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformer_conv_attention.data_loading.processors.time_series_processor import TimeSeriesProcessor
from transformer_conv_attention.utils.exceptions import DataError

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
        'value': np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 0.1, len(dates)),
        'feature1': np.random.normal(0, 1, len(dates))
    })

    return df

def test_processor_initialization():
    processor = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp',
        scaling_method='standard'
    )
    assert not processor.fitted
    assert processor.target_column == 'value'
    assert processor.timestamp_column == 'timestamp'

def test_processor_fit(sample_data):
    processor = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp'
    )

    processor.fit(sample_data)
    assert processor.fitted
    assert 'value' in processor.scalers
    assert 'feature1' in processor.scalers

def test_processor_transform(sample_data):
    processor = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp'
    )

    processor.fit(sample_data)
    transformed_data = processor.transform(sample_data)

    # Check if time features were added
    assert 'hour_sin' in transformed_data.columns
    assert 'hour_cos' in transformed_data.columns
    assert 'day_sin' in transformed_data.columns
    assert 'day_cos' in transformed_data.columns

    # Check if data was scaled
    assert abs(transformed_data['value'].mean()) < 0.1  # Should be close to 0 for standard scaling
    assert abs(transformed_data['value'].std() - 1.0) < 0.1  # Should be close to 1

def test_processor_save_load(sample_data, tmp_path):
    processor = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp'
    )

    processor.fit(sample_data)
    save_path = tmp_path / "processor.pkl"

    # Save and load
    processor.save(save_path)

    new_processor = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp'
    )
    new_processor.load(save_path)

    # Check if states match
    assert processor.scalers == new_processor.scalers
    assert processor.fitted == new_processor.fitted

def test_invalid_scaling_method():
    """Test that invalid scaling method raises ValueError"""
    with pytest.raises(ValueError, match="Invalid scaling method"):
        TimeSeriesProcessor(
            target_column='value',
            timestamp_column='timestamp',
            scaling_method='invalid'
        )

def test_valid_scaling_methods():
    """Test that valid scaling methods are accepted"""
    # Test standard scaling
    processor1 = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp',
        scaling_method='standard'
    )
    assert processor1.scaling_method == 'standard'

    # Test minmax scaling
    processor2 = TimeSeriesProcessor(
        target_column='value',
        timestamp_column='timestamp',
        scaling_method='minmax'
    )
    assert processor2.scaling_method == 'minmax'

def test_missing_columns(sample_data):
    processor = TimeSeriesProcessor(
        target_column='non_existent',
        timestamp_column='timestamp'
    )

    with pytest.raises(DataError):
        processor.fit(sample_data)