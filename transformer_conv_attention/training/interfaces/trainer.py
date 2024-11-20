from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    """Abstract training strategy"""

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def validate_step(self, batch):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass