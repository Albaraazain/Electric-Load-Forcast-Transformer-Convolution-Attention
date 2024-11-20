# transformer_conv_attention/config/data_config.py
from dataclasses import dataclass
from typing import Optional, List
from .base_config import BaseConfig

@dataclass
class DataConfig(BaseConfig):
    """Configuration for data loading"""
    data_path: str
    target_column: str
    timestamp_column: str
    feature_columns: Optional[List[str]] = None
    window_size: int = 168
    prediction_horizon: int = 24
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    stride: int = 1
    scaling_method: str = 'standard'

    def validate(self) -> None:
        """Validate configuration values"""
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if self.val_ratio <= 0 or self.val_ratio >= 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")