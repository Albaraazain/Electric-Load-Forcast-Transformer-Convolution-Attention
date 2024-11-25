# tests/main_pipeline/test_pipeline_builder.py
import pytest
from transformer_conv_attention.main_pipeline.pipeline_builder import TransformerPipelineBuilder
from transformer_conv_attention.main_pipeline.pipeline_director import PipelineDirector
from transformer_conv_attention.main_pipeline.stages import (
    DataLoadingStage,
    ModelSetupStage,
    TrainingStage,
    EvaluationStage
)

def test_pipeline_builder_creates_stages():
    builder = TransformerPipelineBuilder()

    # Test individual stage creation
    builder.add_data_loading_stage()
    pipeline = builder.get_pipeline()
    assert len(pipeline) == 1
    assert isinstance(pipeline[0], DataLoadingStage)

    # Test full pipeline creation
    director = PipelineDirector(builder)
    pipeline = director.build_training_pipeline()

    assert len(pipeline) == 4
    assert isinstance(pipeline[0], DataLoadingStage)
    assert isinstance(pipeline[1], ModelSetupStage)
    assert isinstance(pipeline[2], TrainingStage)
    assert isinstance(pipeline[3], EvaluationStage)

def test_pipeline_config_validation():
    from transformer_conv_attention.main_pipeline.pipeline_config import PipelineConfig

    # Test valid config
    valid_config = PipelineConfig(
        data_path="data/train.csv",
        batch_size=32,
        num_workers=4,
        model_type="transformer",
        max_epochs=100,
        learning_rate=0.001,
        early_stopping_patience=10,
        eval_metrics=["mape", "rmse"]
    )
    valid_config.validate()  # Should not raise

    # Test invalid config
    with pytest.raises(ValueError):
        invalid_config = PipelineConfig(
            data_path="",  # Invalid
            batch_size=32,
            num_workers=4,
            model_type="transformer",
            max_epochs=100,
            learning_rate=0.001,
            early_stopping_patience=10,
            eval_metrics=["mape", "rmse"]
        )
        invalid_config.validate()