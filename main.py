from config.model_config import TransformerConfig
from transformer_conv_attention.models import ModelFactory
from training.strategies.transformer_trainer import TransformerTrainer
from data_loading.loaders.time_series_loader import TimeSeriesLoader

def main():
    # Load configuration
    config = TransformerConfig.from_dict({
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'kernel_size': 3,
        'max_seq_length': 1000,
        'learning_rate': 0.0001,
        'batch_size': 32
    })

    # Create model using factory
    model = ModelFactory.create_model('conv_transformer', config)

    # Setup data loading
    data_loader = TimeSeriesLoader(config)
    train_dataset, val_dataset, test_dataset = data_loader.load_train_val_test()

    # Setup training
    trainer = TransformerTrainer(
        model=model,
        config=config,
        callbacks=[
            EarlyStoppingCallback(patience=5),
            ModelCheckpointCallback(save_dir='checkpoints')
        ]
    )

    # Train model
    trainer.train(train_dataset, val_dataset)

    # Evaluate model
    evaluator = RegressionEvaluator(metrics=['mape', 'rmse'])
    results = evaluator.evaluate(model, test_dataset)

if __name__ == "__main__":
    main()