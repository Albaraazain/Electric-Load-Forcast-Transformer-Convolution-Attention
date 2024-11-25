# transformer_conv_attention/main_pipeline/pipeline_director.py
from typing import List
from .interfaces.pipeline_builder import PipelineBuilder
from .interfaces.pipeline_stage import PipelineStage
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PipelineDirector:
    """Director for coordinating pipeline construction"""

    def __init__(self, builder: PipelineBuilder):
        self._builder = builder

    def build_training_pipeline(self) -> List[PipelineStage]:
        """Build complete training pipeline"""
        logger.info("Building training pipeline")
        self._builder.reset()
        self._builder.add_data_loading_stage()
        self._builder.add_model_setup_stage()
        self._builder.add_training_stage()
        self._builder.add_evaluation_stage()
        return self._builder.get_pipeline()

    def build_inference_pipeline(self) -> List[PipelineStage]:
        """Build inference pipeline"""
        logger.info("Building inference pipeline")
        self._builder.reset()
        self._builder.add_data_loading_stage()
        self._builder.add_model_setup_stage()
        self._builder.add_evaluation_stage()
        return self._builder.get_pipeline()