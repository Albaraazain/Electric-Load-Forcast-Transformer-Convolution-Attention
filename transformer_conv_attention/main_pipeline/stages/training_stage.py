# transformer_conv_attention/main_pipeline/stages/training_stage.py
import torch.nn as nn
from torch.optim import AdamW
from ...training.time_series_trainer import TimeSeriesTrainer
from .base_stage import BaseStage

class TrainingStage(BaseStage):
    """Stage for model training"""

    def __init__(self):
        super().__init__("Training")

    def _execute(self) -> None:
        # Get required components from context
        model = self.context.get_data('model')
        train_loader = self.context.get_data('train_loader')
        val_loader = self.context.get_data('val_loader')
        config = self.context.get_data('config')
        device = self.context.get_data('device')

        # Setup training components
        criterion = nn.L1Loss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize trainer
        trainer = TimeSeriesTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device
        )

        # Train model
        history = trainer.train()

        # Save results to context
        self.context.set_data('training_history', history)
        self.context.set_data('trained_model', model)

    def validate(self) -> bool:
        required_data = ['model', 'train_loader', 'val_loader', 'config', 'device']
        for item in required_data:
            if not self.context.get_data(item):
                raise ValueError(f"Missing required context data: {item}")
        return True