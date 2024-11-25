# transformer_conv_attention/config/time_series_config.py
from dataclasses import dataclass
from typing import List
from .model_config import TransformerConfig
from ..utils.exceptions import ConfigurationError

@dataclass
class TimeSeriesConfig(TransformerConfig):
    """Configuration for time series transformer"""
    # Time series specific fields
    input_size: int
    output_size: int
    sequence_length: int
    prediction_length: int
    input_features: List[str]
    target_features: List[str]
    scaling_method: str = 'standard'

    def __post_init__(self):
        """Validate configuration values after initialization"""
        print(f"[DEBUG] TimeSeriesConfig initialized with:")
        print(f"[DEBUG] - d_model: {self.d_model}")
        print(f"[DEBUG] - sequence_length: {self.sequence_length}")
        print(f"[DEBUG] - prediction_length: {self.prediction_length}")
        print(f"[DEBUG] - input_features: {self.input_features}")

        # Validate both transformer and time series configs
        super().validate()  # Call TransformerConfig validation
        self.validate()     # Call local validation

    def validate(self) -> None:
        """Validate time series specific configuration values"""
        if self.input_size <= 0:
            raise ConfigurationError("input_size must be positive")
        if self.output_size <= 0:
            raise ConfigurationError("output_size must be positive")
        if self.sequence_length <= 0:
            raise ConfigurationError("sequence_length must be positive")
        if self.prediction_length <= 0:
            raise ConfigurationError("prediction_length must be positive")
        if not self.input_features:
            raise ConfigurationError("input_features cannot be empty")
        if not self.target_features:
            raise ConfigurationError("target_features cannot be empty")
        if self.scaling_method not in ['standard', 'minmax']:
            raise ConfigurationError("scaling_method must be 'standard' or 'minmax'")