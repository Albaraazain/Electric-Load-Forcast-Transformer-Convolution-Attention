# tests/integration/test_model_to_training.py
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from transformer_conv_attention.config import TrainingConfig
from transformer_conv_attention.training.time_series_trainer import TimeSeriesTrainer
from transformer_conv_attention.models import ModelFactory

class TestModelToTrainingIntegration:
    @pytest.fixture
    def training_config(self):
        return TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            max_epochs=2,
            patience=5,
            window_size=168,
            prediction_horizon=24,
            validation_split=0.2,
            teacher_forcing_ratio=0.5
        )

    def test_model_training_integration(self, config, training_config):
        """Test model integration with training pipeline"""
        # Create model
        model = ModelFactory.create_model('time_series_transformer', config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Setup training components
        criterion = nn.L1Loss()
        optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)

        # Create mock data
        train_data = create_mock_training_data(32, config.sequence_length, config.input_size)
        val_data = create_mock_training_data(16, config.sequence_length, config.input_size)

        trainer = TimeSeriesTrainer(
            model=model,
            train_loader=train_data,
            val_loader=val_data,
            criterion=criterion,
            optimizer=optimizer,
            config=training_config,
            device=device
        )

        # Train for a few epochs
        history = trainer.train()

        # Verify training results
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
        assert history['train_loss'][-1] < history['train_loss'][0]  # Loss decreased

    def test_gradient_flow(self, config, training_config):
        """Test gradient flow through model during training"""
        model = ModelFactory.create_model('time_series_transformer', config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Setup training
        criterion = nn.L1Loss()
        optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)

        # Single batch training step
        x = torch.randn(4, config.sequence_length, config.input_size).to(device)
        y = torch.randn(4, config.prediction_length, 1).to(device)

        optimizer.zero_grad()
        predictions, _ = model(x, config.prediction_length)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        for name, param in model.named_parameters():
            assert not torch.equal(param, initial_params[name])

    def test_model_checkpointing(self, config, training_config, tmp_path):
        """Test model checkpointing during training"""
        model = ModelFactory.create_model('time_series_transformer', config)

        # Save initial state
        checkpoint_path = tmp_path / "model_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, checkpoint_path)

        # Create new model and load checkpoint
        new_model = ModelFactory.create_model('time_series_transformer', config)
        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint['model_state_dict'])

        # Verify models are identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

def create_mock_training_data(num_batches, seq_length, input_size):
    """Create mock training data loader"""
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(num_batches * 4, seq_length, input_size)
    y = torch.randn(num_batches * 4, 24, 1)  # 24 is prediction horizon

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4, shuffle=True)