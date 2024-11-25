# transformer_conv_attention/main_pipeline/stages/evaluation_stage.py
from ...evaluation.evaluator import TimeSeriesEvaluator
from .base_stage import BaseStage

class EvaluationStage(BaseStage):
    """Stage for model evaluation"""

    def __init__(self):
        super().__init__("Evaluation")

    def _execute(self) -> None:
        # Get components from context
        model = self.context.get_data('trained_model')
        test_loader = self.context.get_data('test_loader')
        config = self.context.get_data('config')
        device = self.context.get_data('device')
        processor = self.context.get_data('data_processor')

        # Initialize evaluator
        evaluator = TimeSeriesEvaluator(
            model=model,
            device=device,
            metrics=config.eval_metrics,
            scaler=processor
        )

        # Evaluate model
        metrics, attention_weights = evaluator.evaluate(
            test_loader,
            prediction_length=config.prediction_horizon
        )

        # Save results to context
        self.context.set_data('evaluation_metrics', metrics)
        self.context.set_data('attention_weights', attention_weights)

    def validate(self) -> bool:
        required_data = ['trained_model', 'test_loader', 'config', 'device', 'data_processor']
        for item in required_data:
            if not self.context.get_data(item):
                raise ValueError(f"Missing required context data: {item}")
        return True