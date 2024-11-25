# transformer_conv_attention/main_pipeline/pipeline_builder.py
from typing import List, Optional
from .interfaces.pipeline_builder import PipelineBuilder
from .interfaces.pipeline_stage import PipelineStage
from .stages.data_loading_stage import DataLoadingStage
from .stages.model_setup_stage import ModelSetupStage
from .stages.training_stage import TrainingStage
from .stages.evaluation_stage import EvaluationStage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TransformerPipelineBuilder(PipelineBuilder):
    """Concrete builder for transformer pipeline"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._pipeline: List[PipelineStage] = []

    def add_data_loading_stage(self) -> None:
        logger.info("Adding data loading stage")
        stage = DataLoadingStage()
        self._pipeline.append(stage)

    def add_model_setup_stage(self) -> None:
        logger.info("Adding model setup stage")
        stage = ModelSetupStage()
        self._pipeline.append(stage)

    def add_training_stage(self) -> None:
        logger.info("Adding training stage")
        stage = TrainingStage()
        self._pipeline.append(stage)

    def add_evaluation_stage(self) -> None:
        logger.info("Adding evaluation stage")
        stage = EvaluationStage()
        self._pipeline.append(stage)

    def get_pipeline(self) -> List[PipelineStage]:
        pipeline = self._pipeline
        self.reset()
        return pipeline