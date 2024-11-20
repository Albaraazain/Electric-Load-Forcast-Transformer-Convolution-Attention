# transformer_conv_attention/data_loading/interfaces/processor_interface.py
from abc import ABC, abstractmethod
import pandas as pd
import torch
from typing import Tuple, Dict, Any

class DataProcessor(ABC):
    """Abstract base class for data processors"""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit processor to data"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save processor state"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load processor state"""
        pass