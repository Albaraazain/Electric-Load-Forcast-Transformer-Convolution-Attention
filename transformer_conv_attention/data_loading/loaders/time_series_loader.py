# transformer_conv_attention/data_loading/loaders/time_series_loader.py
from typing import Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from ..processors.time_series_processor import TimeSeriesProcessor
from ..datasets.time_series_dataset import TransformerTimeSeriesDataset
from ...utils.logger import get_logger
from ...utils.exceptions import DataError

logger = get_logger(__name__)

class TimeSeriesLoader:
    """Loader for time series data"""

    def __init__(
            self,
            data_path: str,
            target_column: str,
            timestamp_column: str,
            feature_columns: Optional[list] = None,
            window_size: int = 168,  # 1 week for hourly data
            prediction_horizon: int = 24,  # 1 day ahead
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            batch_size: int = 32,
            stride: int = 1,
            scaling_method: str = 'standard'
    ):
        """
        Initialize loader

        Args:
            data_path: Path to data file
            target_column: Name of target column
            timestamp_column: Name of timestamp column
            feature_columns: List of feature columns (optional)
            window_size: Size of input window
            prediction_horizon: Number of future steps to predict
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            batch_size: Batch size for DataLoader
            stride: Stride for sliding window
            scaling_method: Method for scaling data
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.feature_columns = feature_columns or []
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.stride = stride

        # Initialize processor
        self.processor = TimeSeriesProcessor(
            target_column=target_column,
            timestamp_column=timestamp_column,
            scaling_method=scaling_method
        )

        print(f"[DEBUG] Initializing TimeSeriesLoader with:")
        print(f"[DEBUG] - data_path: {data_path}")
        print(f"[DEBUG] - window_size: {window_size}")
        print(f"[DEBUG] - prediction_horizon: {prediction_horizon}")

        # Load and process data
        self._load_data()

        print(f"[DEBUG] Initial data shape: {self.data.shape}")
        print(f"[DEBUG] Data columns: {self.data.columns.tolist()}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the existing processor"""
        return self.processor.transform(data)

    def _load_data(self) -> None:
        """Load and preprocess data"""
        try:
            # Load data
            if self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == '.parquet':
                data = pd.read_parquet(self.data_path)
            else:
                raise DataError(f"Unsupported file format: {self.data_path.suffix}")

            # Process data
            self.data = self.processor.fit_transform(data)

            # Calculate minimum size needed for each split
            min_split_size = self.window_size + self.prediction_horizon
            total_min_size = min_split_size * 3  # Need enough for train, val, and test

            if len(self.data) < total_min_size:
                raise DataError(
                    f"Data length ({len(self.data)}) must be at least "
                    f"3 * (window_size + prediction_horizon) = {total_min_size}"
                )

            # Adjust ratios to ensure minimum sizes
            available_data = len(self.data)

            # Calculate sizes ensuring each split has at least min_split_size
            train_size = max(
                int(available_data * self.train_ratio),
                min_split_size
            )
            val_size = max(
                int(available_data * self.val_ratio),
                min_split_size
            )
            test_size = max(
                available_data - train_size - val_size,
                min_split_size
            )

            # Recompute train_size if necessary
            if train_size + val_size + test_size > available_data:
                excess = (train_size + val_size + test_size) - available_data
                train_size -= excess

            print(f"[DEBUG] Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

            self.train_data = self.data[:train_size]
            self.val_data = self.data[train_size:train_size + val_size]
            self.test_data = self.data[train_size + val_size:]

            logger.info(f"Data loaded and processed: {len(self.data)} samples")
            logger.info(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise DataError(f"Failed to load data: {str(e)}")

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test dataloaders"""

        print(f"[DEBUG] Starting get_dataloaders()")
        print(f"[DEBUG] Data shape before splitting: {self.data.shape}")




        # Create datasets
        train_dataset = TransformerTimeSeriesDataset(
            self.train_data,
            self.window_size,
            self.prediction_horizon,
            self.target_column,
            self.timestamp_column,
            self.feature_columns,
            self.stride
        )

        val_dataset = TransformerTimeSeriesDataset(
            self.val_data,
            self.window_size,
            self.prediction_horizon,
            self.target_column,
            self.timestamp_column,
            self.feature_columns,
            stride=1  # Use stride=1 for validation
        )

        test_dataset = TransformerTimeSeriesDataset(
            self.test_data,
            self.window_size,
            self.prediction_horizon,
            self.target_column,
            self.timestamp_column,
            self.feature_columns,
            stride=1  # Use stride=1 for testing
        )
        print(f"[DEBUG] Dataset shapes:")
        print(f"[DEBUG] - Train: {train_dataset.shape} (samples, window_size, features)")
        print(f"[DEBUG] - Val: {val_dataset.shape} (samples, window_size, features)")
        print(f"[DEBUG] - Test: {test_dataset.shape} (samples, window_size, features)")



        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def save_processor(self, path: str) -> None:
        """Save processor state"""
        self.processor.save(path)

    def load_processor(self, path: str) -> None:
        """Load processor state"""
        self.processor.load(path)