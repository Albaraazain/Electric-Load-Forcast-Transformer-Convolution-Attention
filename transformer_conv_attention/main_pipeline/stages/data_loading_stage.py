# transformer_conv_attention/main_pipeline/stages/data_loading_stage.py
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from ...data_loading.loaders.time_series_loader import TimeSeriesLoader
from .base_stage import BaseStage

class DataLoadingStage(BaseStage):
    """Stage for data loading and preprocessing"""

    def __init__(self):
        super().__init__("DataLoading")

    def _execute(self) -> None:
        config = self.context.get_data('config')

        # Initialize data loader
        loader = TimeSeriesLoader(
            data_path=config.data_path,
            target_column=config.target_column,
            timestamp_column=config.timestamp_column,
            feature_columns=config.feature_columns,
            window_size=config.window_size,
            prediction_horizon=config.prediction_horizon
        )

        # Get data loaders
        train_loader, val_loader, test_loader = loader.get_dataloaders()

        # Save to context
        self.context.set_data('train_loader', train_loader)
        self.context.set_data('val_loader', val_loader)
        self.context.set_data('test_loader', test_loader)
        self.context.set_data('data_processor', loader.processor)

    def validate(self) -> bool:
        config = self.context.get_data('config')
        if not config:
            raise ValueError("Configuration not found in context")
        required_fields = ['data_path', 'target_column', 'timestamp_column']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required config field: {field}")
        return True