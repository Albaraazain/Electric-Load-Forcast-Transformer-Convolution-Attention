# transformer_conv_attention/config/time_series_config.py
from dataclasses import dataclass
from typing import List, Optional
from .model_config import TransformerConfig

@dataclass
class TimeSeriesConfig(TransformerConfig):
    """Configuration for time series transformer"""
    input_size: int
    output_size: int
    sequence_length: int
    prediction_length: int
    input_features: List[str]
    target_features: List[str]
    scaling_method: str = 'standard'

    def __post_init__(self):
        """Validate configuration values after initialization"""
        super().validate()
        self.validate()

    def validate(self) -> None:
        """Validate configuration values"""
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if not self.input_features:
            raise ValueError("input_features cannot be empty")
        if not self.target_features:
            raise ValueError("target_features cannot be empty")
        if self.scaling_method not in ['standard', 'minmax']:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")