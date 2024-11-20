from abc import ABC, abstractmethod

class TrainingCallback(ABC):
    """Base callback for training events"""

    @abstractmethod
    def on_epoch_start(self, trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer):
        pass

    @abstractmethod
    def on_batch_start(self, trainer, batch):
        pass

    @abstractmethod
    def on_batch_end(self, trainer, batch, outputs):
        pass