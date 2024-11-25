# transformer_conv_attention/main_pipeline/stages/base_stage.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..interfaces.pipeline_stage import PipelineStage, PipelineContext
from ...utils.logger import get_logger
from ...utils.exceptions import StageError

logger = get_logger(__name__)

class BaseStage(PipelineStage):
    """Base class for pipeline stages"""

    def __init__(self, name: str):
        self.name = name
        self.context: Optional[PipelineContext] = None

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute stage with error handling and logging"""
        logger.info(f"Executing stage: {self.name}")
        self.context = context

        try:
            self.validate()
            self._execute()
            logger.info(f"Successfully completed stage: {self.name}")
            return self.context
        except Exception as e:
            logger.error(f"Error in stage {self.name}: {str(e)}")
            raise StageError(f"Stage {self.name} failed: {str(e)}") from e

    def validate(self) -> bool:
        """Base validation - can be overridden by subclasses"""
        return True

    @abstractmethod
    def _execute(self) -> None:
        """Concrete stage implementation"""
        pass