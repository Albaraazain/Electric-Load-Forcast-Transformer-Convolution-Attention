# transformer_conv_attention/prediction/time_series_predictor.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesPredictor:
    """Handler for time series predictions"""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            scaler: Optional[object] = None
    ):
        self.model = model
        self.device = device
        self.scaler = scaler
        self.model.eval()

    def predict(
            self,
            x: torch.Tensor,
            prediction_length: int,
            batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions

        Args:
            x: Input tensor [batch_size, sequence_length, features]
            prediction_length: Number of steps to predict
            batch_size: Optional batch size for processing

        Returns:
            - Predictions array
            - Dictionary of attention weights
        """
        if batch_size is None:
            return self._predict_batch(x, prediction_length)

        # Process in batches
        predictions = []
        attention_weights = []

        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_preds, batch_weights = self._predict_batch(
                batch_x,
                prediction_length
            )
            predictions.append(batch_preds)
            attention_weights.append(batch_weights)

        # Combine results
        predictions = np.concatenate(predictions, axis=0)
        combined_weights = self._combine_attention_weights(attention_weights)

        return predictions, combined_weights

    def _predict_batch(
            self,
            x: torch.Tensor,
            prediction_length: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate predictions for a single batch"""
        x = x.to(self.device)

        with torch.no_grad():
            predictions, attention_weights = self.model(
                x,
                target_len=prediction_length
            )

            # Move to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            attention_weights = {
                k: v.cpu().numpy() for k, v in attention_weights.items()
            }

            # Inverse transform if scaler exists
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(predictions)

        return predictions, attention_weights

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