# transformer_conv_attention/main_pipeline/interfaces/pipeline_builder.py
from abc import ABC, abstractmethod
from typing import List
from .pipeline_stage import PipelineStage

class PipelineBuilder(ABC):
    """Interface for pipeline builders"""

    @abstractmethod
    def reset(self) -> None:
        """Reset builder to initial state"""
        pass

    @abstractmethod
    def add_data_loading_stage(self) -> None:
        """Add data loading stage to pipeline"""
        pass

    @abstractmethod
    def add_model_setup_stage(self) -> None:
        """Add model setup stage to pipeline"""
        pass

    @abstractmethod
    def add_training_stage(self) -> None:
        """Add training stage to pipeline"""
        pass

    @abstractmethod
    def add_evaluation_stage(self) -> None:
        """Add evaluation stage to pipeline"""
        pass

    @abstractmethod
    def get_pipeline(self) -> List[PipelineStage]:
        """Get built pipeline"""
        pass