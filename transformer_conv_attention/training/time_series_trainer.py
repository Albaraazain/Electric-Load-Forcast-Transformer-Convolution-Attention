# transformer_conv_attention/training/time_series_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .trainer import Trainer  # Updated import

from .time_series_training_config import TimeSeriesTrainingConfig
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
            config: 'TimeSeriesTrainingConfig',
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
            min_lr=config.early_stopping_min_delta,
            verbose=True
        )

    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for x, target in self.train_loader:
            x = x.to(self.device)
            target = target.to(self.device)

            # Ensure tensors require grad
            x.requires_grad_(True)
            target.requires_grad_(True)

            self.optimizer.zero_grad()

            # Forward pass
            predictions, _ = self.model(x, self.config.prediction_horizon)

            # Reshape predictions and target if needed
            if predictions.size() != target.size():
                predictions = predictions.view(target.size())

            # Calculate loss
            loss = self.criterion(predictions, target)

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

    # transformer_conv_attention/training/time_series_trainer.py

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
                f'Epoch {epoch + 1}/{self.config.max_epochs} - '
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
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f'Early stopping triggered after {epoch + 1} epochs'
                )
                break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        return history
