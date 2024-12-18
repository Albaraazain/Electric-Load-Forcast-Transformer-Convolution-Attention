from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from .interfaces.evaluator_interface import EvaluatorInterface
from .factory.metric_factory import MetricFactory
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesEvaluator(EvaluatorInterface):
    """Evaluator for time series models"""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            metrics: Optional[List[str]] = None,
            scaler: Optional[object] = None
    ):
        self.model = model
        self.device = device
        self.scaler = scaler
        self.metrics = []
        self.model.eval()

        # Initialize metrics
        if metrics:
            for metric in metrics:
                self.add_metric(metric)
        else:
            # Default metrics
            default_metrics = ['mape', 'rmse', 'mae', 'mase']
            for metric in default_metrics:
                self.add_metric(metric)

    def add_metric(self, metric: str) -> None:
        """Add a metric for evaluation"""
        try:
            metric_instance = MetricFactory.create_metric(metric)
            self.metrics.append(metric_instance)
            logger.info(f"Added metric: {metric_instance.get_name()}")
        except ValueError as e:
            logger.error(f"Failed to add metric: {str(e)}")
            raise

    def calculate_metrics(self, predictions: torch.Tensor, actuals: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        for metric in self.metrics:
            metrics[metric.get_name()] = metric.compute(predictions, actuals)
        return metrics

    def evaluate(
            self,
            dataloader: torch.utils.data.DataLoader,
            prediction_length: int
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate model performance"""
        all_predictions = []
        all_actuals = []
        all_attention_weights = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # Generate predictions
                predictions, attention_weights = self.model(
                    x,
                    target_len=prediction_length
                )

                # Move to CPU
                predictions = predictions.cpu()
                actuals = y.cpu()
                attention_weights = {
                    k: v.cpu().numpy() for k, v in attention_weights.items()
                }

                all_predictions.append(predictions)
                all_actuals.append(actuals)
                all_attention_weights.append(attention_weights)

        # Combine results
        predictions = torch.cat(all_predictions, dim=0)
        actuals = torch.cat(all_actuals, dim=0)

        # Inverse transform if scaler exists
        if self.scaler is not None:
            predictions = torch.tensor(self.scaler.inverse_transform(predictions.detach().numpy()))
            actuals = torch.tensor(self.scaler.inverse_transform(actuals.detach().numpy()))

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, actuals)

        # Combine attention weights
        combined_attention_weights = {
            k: np.concatenate([aw[k] for aw in all_attention_weights], axis=0)
            for k in all_attention_weights[0]
        }

        return metrics, combined_attention_weights

    def _combine_attention_weights(
            self,
            attention_weights_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Combine attention weights from multiple batches"""
        combined = {}
        for key in attention_weights_list[0].keys():
            combined[key] = np.concatenate(
                [weights[key] for weights in attention_weights_list],
                axis=0
            )
        return combined