# transformer_conv_attention/training/time_series_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .trainer import Trainer
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesTrainer(Trainer):
    """Trainer specialized for time series forecasting"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            config: TimeSeriesTrainingConfig,
            device: torch.device
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.learning_rate_scheduler_factor,
            patience=config.learning_rate_scheduler_patience,
            min_delta=config.early_stopping_min_delta,
            verbose=True
        )

        # Training metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions, _ = self.model(
                x,
                target_len=self.config.prediction_horizon
            )

            # Calculate loss
            loss = self.criterion(predictions, y)

            # Backward pass
            loss.backward()

            # Gradient clipping if configured
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f'Train Batch {batch_idx}/{len(self.train_loader)} '
                    f'Loss: {loss.item():.6f}'
                )

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                predictions, _ = self.model(
                    x,
                    target_len=self.config.prediction_horizon
                )

                # Calculate loss
                loss = self.criterion(predictions, y)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> Dict[str, list]:
        """
        Train the model

        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        for epoch in range(self.config.max_epochs):
            # Train epoch
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            logger.info(
                f'Epoch {epoch+1}/{self.config.max_epochs} - '
                f'Train Loss: {train_loss:.6f} - '
                f'Val Loss: {val_loss:.6f}'
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(
                    f'Early stopping triggered after {epoch+1} epochs'
                )
                break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        return history