# transformer_conv_attention/config/base_config.py
from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BaseConfig:
    """Base configuration class with common functionality"""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, save_path: str) -> None:
        """Save configuration to YAML file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to {save_path}")
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    def validate(self) -> None:
        """Validate configuration values"""
        pass