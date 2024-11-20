from abc import ABC, abstractmethod
from typing import Tuple
import torch.utils.data as data

class DatasetLoader(ABC):
    """Abstract dataset loader"""

    @abstractmethod
    def load_train_val_test(self) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
        pass

    @abstractmethod
    def get_data_loader(self, dataset: data.Dataset, batch_size: int):
        pass