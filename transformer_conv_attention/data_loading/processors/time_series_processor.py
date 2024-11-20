# transformer_conv_attention/data_loading/processors/time_series_processor.py
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any
import pickle
from pathlib import Path

from ..interfaces.processor_interface import DataProcessor
from ...utils.logger import get_logger
from ...utils.exceptions import DataError

logger = get_logger(__name__)

class TimeSeriesProcessor(DataProcessor):
    """Processor for time series data"""

    VALID_SCALING_METHODS = ['standard', 'minmax']  # Add class variable for valid methods

    def __init__(self,
                 target_column: str,
                 timestamp_column: str,
                 scaling_method: str = 'standard'):
        """
        Initialize processor

        Args:
            target_column: Name of target column
            timestamp_column: Name of timestamp column
            scaling_method: Scaling method ('standard' or 'minmax')

        Raises:
            ValueError: If scaling_method is not valid
        """
        if scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(
                f"Invalid scaling method: {scaling_method}. "
                f"Must be one of {self.VALID_SCALING_METHODS}"
            )

        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.scaling_method = scaling_method
        self.scalers = {}
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit processor to data

        Args:
            data: Input DataFrame
        """
        try:
            # Validate columns
            self._validate_columns(data)

            # Fit scalers
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.scaling_method == 'standard':
                    mean = data[col].mean()
                    std = data[col].std()
                    self.scalers[col] = {'mean': mean, 'std': std}
                elif self.scaling_method == 'minmax':
                    min_val = data[col].min()
                    max_val = data[col].max()
                    self.scalers[col] = {'min': min_val, 'max': max_val}
                else:
                    raise ValueError(f"Unknown scaling method: {self.scaling_method}")

            # Add time features
            self._extract_time_features(data)

            self.fitted = True
            logger.info("Processor fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting processor: {str(e)}")
            raise DataError(f"Failed to fit processor: {str(e)}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise DataError("Processor not fitted. Call fit() first.")

        try:
            # Create copy to avoid modifying original data
            transformed_data = data.copy()

            # Scale numeric columns
            for col, scaler in self.scalers.items():
                if self.scaling_method == 'standard':
                    transformed_data[col] = (data[col] - scaler['mean']) / scaler['std']
                else:  # minmax
                    transformed_data[col] = (data[col] - scaler['min']) / (scaler['max'] - scaler['min'])

            # Add time features
            transformed_data = self._extract_time_features(transformed_data)

            return transformed_data

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise DataError(f"Failed to transform data: {str(e)}")

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform scaled data

        Args:
            data: Scaled tensor

        Returns:
            Unscaled tensor
        """
        if not self.fitted:
            raise DataError("Processor not fitted. Call fit() first.")

        try:
            # Get target scaler
            scaler = self.scalers[self.target_column]

            # Convert to numpy for easier manipulation
            data_np = data.numpy()

            # Inverse transform
            if self.scaling_method == 'standard':
                data_np = data_np * scaler['std'] + scaler['mean']
            else:  # minmax
                data_np = data_np * (scaler['max'] - scaler['min']) + scaler['min']

            return torch.from_numpy(data_np)

        except Exception as e:
            logger.error(f"Error inverse transforming data: {str(e)}")
            raise DataError(f"Failed to inverse transform data: {str(e)}")

    def save(self, path: str) -> None:
        """Save processor state"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'scalers': self.scalers,
            'fitted': self.fitted,
            'target_column': self.target_column,
            'timestamp_column': self.timestamp_column,
            'scaling_method': self.scaling_method
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Processor state saved to {path}")

    def load(self, path: str) -> None:
        """Load processor state"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No processor state found at {path}")

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.scalers = state['scalers']
        self.fitted = state['fitted']
        self.target_column = state['target_column']
        self.timestamp_column = state['timestamp_column']
        self.scaling_method = state['scaling_method']

        logger.info(f"Processor state loaded from {path}")

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate required columns exist"""
        missing_cols = []
        if self.target_column not in data.columns:
            missing_cols.append(self.target_column)
        if self.timestamp_column not in data.columns:
            missing_cols.append(self.timestamp_column)

        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")

    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time features from timestamp column"""
        df = data.copy()
        timestamps = pd.to_datetime(df[self.timestamp_column])

        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)

        return df