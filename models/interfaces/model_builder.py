from abc import ABC, abstractmethod

class ModelBuilder(ABC):
    """Abstract builder for transformer models"""

    @abstractmethod
    def build_attention(self):
        pass

    @abstractmethod
    def build_encoder(self):
        pass

    @abstractmethod
    def build_decoder(self):
        pass

    @abstractmethod
    def build(self, config: dict):
        pass