# transformer_conv_attention/data_loading/interfaces/dataset_interface.py
from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch

class TimeSeriesDataset(ABC, torch.utils.data.Dataset):
    """Abstract base class for time series datasets"""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length"""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        pass