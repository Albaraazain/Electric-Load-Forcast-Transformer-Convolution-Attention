# transformer_conv_attention/evaluation/factory/metric_factory.py
from typing import Dict, Type
from ..interfaces.metric_interface import MetricInterface
from ..metrics.mape_metric import MAPEMetric
from ..metrics.rmse_metric import RMSEMetric

class MetricFactory:
    """Factory for creating metric instances"""

    _metrics: Dict[str, Type[MetricInterface]] = {
        'mape': MAPEMetric,
        'rmse': RMSEMetric,
        # Add other metrics...
    }

    @classmethod
    def create_metric(cls, metric_name: str) -> MetricInterface:
        """Create a metric instance"""
        metric_class = cls._metrics.get(metric_name.lower())
        if metric_class is None:
            raise ValueError(f"Unknown metric: {metric_name}")
        return metric_class()

    @classmethod
    def register_metric(cls, name: str, metric_class: Type[MetricInterface]):
        """Register a new metric"""
        cls._metrics[name.lower()] = metric_class