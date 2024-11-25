# transformer_conv_attention/main_pipeline/stages/model_setup_stage.py
import torch
from ...models import ModelFactory
from .base_stage import BaseStage
from ...models.registery import register_builders


class ModelSetupStage(BaseStage):
    """Stage for model initialization"""

    def __init__(self):
        super().__init__("ModelSetup")
        register_builders()  # Register builders on initialization

    def _execute(self) -> None:
        config = self.context.get_data('config')

        # Create model
        model = ModelFactory.create_model(
            model_type=config.model_type,
            config=config.model_config
        )

        # Load pretrained weights if specified
        if config.model_path:
            model.load_state_dict(torch.load(config.model_path))

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Save to context
        self.context.set_data('model', model)
        self.context.set_data('device', device)

    def validate(self) -> bool:
        config = self.context.get_data('config')
        if not config:
            raise ValueError("Configuration not found in context")
        if not hasattr(config, 'model_type'):
            raise ValueError("model_type not specified in config")
        if not hasattr(config, 'model_config'):
            raise ValueError("model_config not specified in config")
        return True