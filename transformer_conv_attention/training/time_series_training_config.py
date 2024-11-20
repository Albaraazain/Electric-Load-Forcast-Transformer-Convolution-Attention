# transformer_conv_attention/training/time_series_training_config.py
from dataclasses import dataclass
from typing import Optional
from ..config.base_config import BaseConfig

@dataclass
class TimeSeriesTrainingConfig(BaseConfig):
    """Configuration for time series training"""
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    window_size: int
    prediction_horizon: int
    validation_split: float
    teacher_forcing_ratio: float = 0.5
    learning_rate_scheduler_factor: float = 0.1
    learning_rate_scheduler_patience: int = 10
    gradient_clip_val: Optional[float] = None
    early_stopping_min_delta: float = 1e-4

    def validate(self) -> None:
        """Validate configuration values"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")