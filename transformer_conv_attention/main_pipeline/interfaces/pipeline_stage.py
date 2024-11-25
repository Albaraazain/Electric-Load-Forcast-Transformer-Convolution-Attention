# transformer_conv_attention/main_pipeline/interfaces/pipeline_stage.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class PipelineStage(ABC):
    """Base interface for pipeline stages"""

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline stage"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate stage configuration"""
        pass

class PipelineContext:
    """Context for sharing data between pipeline stages"""

    def __init__(self):
        self._data = {}
        self._metadata = {}

    def set_data(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get_data(self, key: str) -> Any:
        return self._data.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self._metadata.get(key)