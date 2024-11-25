# transformer_conv_attention/main_pipeline/pipeline_config.py
from dataclasses import dataclass
from typing import Optional, List
from ..config.base_config import BaseConfig

@dataclass
class PipelineConfig(BaseConfig):
    """Configuration for pipeline execution"""

    # Required fields (no defaults)
    data_path: str
    target_column: str
    timestamp_column: str
    feature_columns: List[str]
    batch_size: int
    num_workers: int
    model_type: str
    model_config: dict
    max_epochs: int
    learning_rate: float
    early_stopping_patience: int
    eval_metrics: List[str]
    window_size: int  # Added
    prediction_horizon: int  # Added

    # Optional fields (with defaults)
    model_path: Optional[str] = None
    weight_decay: float = 0.01
    gradient_clip_val: Optional[float] = None
    learning_rate_scheduler_factor: float = 0.1
    learning_rate_scheduler_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    save_predictions: bool = True

    def validate(self) -> None:
        """Validate configuration values"""
        if not self.data_path:
            raise ValueError("data_path must be provided")
        if not self.target_column:
            raise ValueError("target_column must be provided")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be provided")
        if not self.feature_columns:
            raise ValueError("feature_columns must be provided")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not self.eval_metrics:
            raise ValueError("eval_metrics cannot be empty")