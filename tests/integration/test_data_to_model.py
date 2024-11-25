# tests/integration/test_data_to_model.py
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformer_conv_attention.data_loading.loaders.time_series_loader import TimeSeriesLoader
from transformer_conv_attention.models import ModelFactory
from transformer_conv_attention.config.time_series_config import TimeSeriesConfig
from transformer_conv_attention.utils.logger import get_logger

logger = get_logger(__name__)

class TestDataToModelIntegration:
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample time series data"""
        # Create more comprehensive test data
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')

        # Create sinusoidal pattern with noise for more realistic data
        t = np.linspace(0, 8*np.pi, n_samples)
        base_signal = np.sin(t) + 0.5 * np.sin(2*t) + 0.2 * np.sin(4*t)
        noise = np.random.normal(0, 0.1, n_samples)

        df = pd.DataFrame({
            'timestamp': dates,
            'target': base_signal + noise,
            'feature1': np.sin(t/2) + np.random.normal(0, 0.1, n_samples),
            'feature2': np.cos(t/3) + np.random.normal(0, 0.1, n_samples)
        })

        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * dates.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * dates.hour/24)
        df['day_sin'] = np.sin(2 * np.pi * dates.dayofweek/7)
        df['day_cos'] = np.cos(2 * np.pi * dates.dayofweek/7)

        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        logger.info(f"Generated sample data with {n_samples} data points.")
        return data_path

    @pytest.fixture
    def config(self):
        return TimeSeriesConfig(
            # Model architecture
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            kernel_size=3,
            max_seq_length=1000,
            dropout=0.1,

            # Data configuration
            input_size=12,  # target + 2 features + 4 time features
            output_size=1,
            sequence_length=168,  # 1 week
            prediction_length=24,  # 1 day

            # Features
            input_features=['feature1', 'feature2'],
            target_features=['target'],
            scaling_method='standard'
        )

    def test_data_loading_to_model_dimensions(self, sample_data, config):
        """Test that data dimensions match model expectations"""
        print("\n[DEBUG] ====== Starting Data Loading Test ======")
        print(f"[DEBUG] Test configuration:")
        print(f"[DEBUG] - Sample data path: {sample_data}")
        print(f"[DEBUG] - Config sequence_length: {config.sequence_length}")
        print(f"[DEBUG] - Config prediction_length: {config.prediction_length}")
        print(f"[DEBUG] - Config input_size: {config.input_size}")
        print(f"[DEBUG] - Config output_size: {config.output_size}")

        # Load data
        loader = TimeSeriesLoader(
            data_path=str(sample_data),
            target_column='target',
            timestamp_column='timestamp',
            feature_columns=['feature1', 'feature2',
                             'hour_sin', 'hour_cos',
                             'day_sin', 'day_cos'],
            window_size=config.sequence_length,
            prediction_horizon=config.prediction_length
        )

        print("\n[DEBUG] Getting dataloaders...")
        train_loader, val_loader, _ = loader.get_dataloaders()

        print("\n[DEBUG] Getting first batch...")
        # Get a batch
        for batch_data in train_loader:
            encoder_input, decoder_input, targets = batch_data
            print(f"[DEBUG] Batch contents:")
            print(f"[DEBUG] - encoder_input shape: {encoder_input.shape}")
            print(f"[DEBUG] - decoder_input shape: {decoder_input.shape}")
            print(f"[DEBUG] - targets shape: {targets.shape}")
            break

        print("\n[DEBUG] Creating model...")
        print(f"[DEBUG] Available model types: {list(ModelFactory._builders.keys())}")
        model = ModelFactory.create_model('time_series_transformer', config)

        print("\n[DEBUG] Preparing inputs for model...")
        # Test forward pass
        encoder_input = encoder_input.to(torch.float32)
        decoder_input = decoder_input.to(torch.float32)

        print(f"[DEBUG] Model input shapes before forward pass:")
        print(f"[DEBUG] - encoder_input: {encoder_input.size()}")
        print(f"[DEBUG] - decoder_input: {decoder_input.size()}")

        print("\n[DEBUG] Running model forward pass...")
        predictions, attention_weights = model(encoder_input, config.prediction_length)

        print("\n[DEBUG] Checking output dimensions...")
        print(f"[DEBUG] - predictions shape: {predictions.shape}")
        print(f"[DEBUG] - encoder_input size: {encoder_input.size()}")
        print(f"[DEBUG] - predictions size: {predictions.size()}")

        # Verify dimensions
        print("\n[DEBUG] Running assertions...")
        try:
            assert encoder_input.size(1) == config.sequence_length, \
                f"Expected sequence length {config.sequence_length}, got {encoder_input.size(1)}"
            assert encoder_input.size(2) == config.input_size, \
                f"Expected input size {config.input_size}, got {encoder_input.size(2)}"
            assert predictions.size(1) == config.prediction_length, \
                f"Expected prediction length {config.prediction_length}, got {predictions.size(1)}"
            assert predictions.size(2) == config.output_size, \
                f"Expected output size {config.output_size}, got {predictions.size(2)}"
            print("[DEBUG] All assertions passed successfully!")
        except AssertionError as e:
            print(f"[DEBUG] Assertion failed: {str(e)}")
            raise

        print("[DEBUG] ====== Test Complete ======\n")


    def test_data_processing_compatibility(self, sample_data, config):
        """Test data processing compatibility with model"""
        loader = TimeSeriesLoader(
            data_path=str(sample_data),
            target_column='target',
            timestamp_column='timestamp',
            feature_columns=['feature1', 'feature2',
                             'hour_sin', 'hour_cos',
                             'day_sin', 'day_cos'],
            window_size=config.sequence_length,
            prediction_horizon=config.prediction_length
        )

        # Test processor state saving and loading
        processor_path = Path("temp_processor.pkl")
        loader.save_processor(processor_path)

        new_loader = TimeSeriesLoader(
            data_path=str(sample_data),
            target_column='target',
            timestamp_column='timestamp',
            feature_columns=['feature1', 'feature2',
                             'hour_sin', 'hour_cos',
                             'day_sin', 'day_cos']
        )
        new_loader.load_processor(processor_path)

        # Clean up
        processor_path.unlink()

        # Compare processed data
        assert loader.processor.scalers.keys() == new_loader.processor.scalers.keys()

    def test_end_to_end_batch_processing(self, sample_data, config):
        """Test end-to-end batch processing through data pipeline and model"""
        # Load and process data
        loader = TimeSeriesLoader(
            data_path=str(sample_data),
            target_column='target',
            timestamp_column='timestamp',
            feature_columns=['feature1', 'feature2',
                             'hour_sin', 'hour_cos',
                             'day_sin', 'day_cos'],
            window_size=config.sequence_length,
            prediction_horizon=config.prediction_length
        )

        train_loader, _, _ = loader.get_dataloaders()

        # Create and prepare model
        model = ModelFactory.create_model('time_series_transformer', config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Process batch
        for batch_data in train_loader:
            encoder_input, decoder_input, targets = batch_data
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)

            # Forward pass
            predictions, attention_weights = model(encoder_input, config.prediction_length)

            # Verify outputs
            assert not torch.isnan(predictions).any()
            assert all(w is not None for w in attention_weights.values())
            break