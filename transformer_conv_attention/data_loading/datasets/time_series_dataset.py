# transformer_conv_attention/data_loading/datasets/time_series_dataset.py
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from ..interfaces.dataset_interface import TimeSeriesDataset
from ...utils.logger import get_logger
from ...utils.exceptions import DataError

logger = get_logger(__name__)

class TransformerTimeSeriesDataset(TimeSeriesDataset):
    """Dataset for transformer time series data"""

    def __init__(
            self,
            data: pd.DataFrame,
            window_size: int,
            prediction_horizon: int,
            target_column: str,
            timestamp_column: str,
            feature_columns: Optional[list] = None,
            stride: int = 1
    ):
        """
        Initialize dataset

        Args:
            data: Input DataFrame
            window_size: Size of input window
            prediction_horizon: Number of future steps to predict
            target_column: Name of target column
            timestamp_column: Name of timestamp column
            feature_columns: List of feature columns (optional)
            stride: Stride for sliding window (default=1)
        """
        self.data = data
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.feature_columns = feature_columns or []
        self.stride = stride

        # Validate inputs
        self._validate_inputs()

        # Prepare data
        self._prepare_data()

        logger.info(
            f"Created dataset with {len(self)} samples, "
            f"window_size={window_size}, "
            f"prediction_horizon={prediction_horizon}"
        )

    def __len__(self) -> int:
        """Get number of samples in dataset"""
        return (len(self.data) - self.window_size - self.prediction_horizon + 1) // self.stride

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
            Tuple of (encoder_input, decoder_input, target)
            - encoder_input: Input sequence for encoder [window_size, n_features]
            - decoder_input: Input sequence for decoder [prediction_horizon, n_features]
            - target: Target sequence [prediction_horizon]
        """
        # Get start and end indices
        start_idx = index * self.stride
        encoder_end_idx = start_idx + self.window_size
        decoder_end_idx = encoder_end_idx + self.prediction_horizon

        # Get input features
        encoder_input = self._get_features(start_idx, encoder_end_idx)
        decoder_input = self._get_features(encoder_end_idx, decoder_end_idx)

        # Get targets
        target = torch.tensor(
            self.data[self.target_column].iloc[encoder_end_idx:decoder_end_idx].values,
            dtype=torch.float32
        )

        return encoder_input, decoder_input, target

    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return len(self.feature_columns) + 6  # +6 for time features

    def _validate_inputs(self) -> None:
        """Validate input parameters"""
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        # Check if required columns exist
        missing_cols = []
        if self.target_column not in self.data.columns:
            missing_cols.append(self.target_column)
        if self.timestamp_column not in self.data.columns:
            missing_cols.append(self.timestamp_column)
        for col in self.feature_columns:
            if col not in self.data.columns:
                missing_cols.append(col)

        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")

        # Check if data is long enough
        min_length = self.window_size + self.prediction_horizon
        if len(self.data) < min_length:
            raise DataError(
                f"Data length ({len(self.data)}) must be at least "
                f"window_size + prediction_horizon ({min_length})"
            )

    def _prepare_data(self) -> None:
        """Prepare data for use"""
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.timestamp_column]):
            self.data[self.timestamp_column] = pd.to_datetime(self.data[self.timestamp_column])

        # Sort by timestamp
        self.data = self.data.sort_values(self.timestamp_column).reset_index(drop=True)

        # Add time features if not present
        if 'hour_sin' not in self.data.columns:
            timestamps = self.data[self.timestamp_column]
            self.data['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            self.data['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            self.data['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            self.data['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
            self.data['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
            self.data['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)

    def _get_features(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """Get features for a given window"""
        # Combine all features
        feature_cols = (
                self.feature_columns +
                ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        )

        features = torch.tensor(
            self.data[feature_cols].iloc[start_idx:end_idx].values,
            dtype=torch.float32
        )

        return features