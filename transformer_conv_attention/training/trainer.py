# transformer_conv_attention/training/trainer.py
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Trainer(ABC):
    """Abstract base class for trainers"""

    def __init__(self):
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    @abstractmethod
    def train_epoch(self) -> float:
        """Train one epoch"""
        pass

    @abstractmethod
    def validate(self) -> float:
        """Validate the model"""
        pass

    @abstractmethod
    def train(self) -> Dict[str, list]:
        """Train the model"""
        pass

    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> None:
        """Save training checkpoint"""
        if not hasattr(self, 'model'):
            raise AttributeError("No model found in trainer")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            **kwargs
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path)

        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint