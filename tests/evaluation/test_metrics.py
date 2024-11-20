# tests/evaluation/test_metrics.py
import pytest
import numpy as np
import torch
from transformer_conv_attention.evaluation.factory.metric_factory import MetricFactory

@pytest.fixture
def sample_data():
    actual = np.array([100, 200, 300, 400, 500])
    predicted = np.array([110, 190, 310, 390, 510])
    return actual, predicted

def test_mape_metric(sample_data):
    actual, predicted = sample_data
    metric = MetricFactory.create_metric('mape')
    value = metric.calculate(actual, predicted)
    assert isinstance(value, float)
    assert 0 <= value <= 100

def test_rmse_metric(sample_data):
    actual, predicted = sample_data
    metric = MetricFactory.create_metric('rmse')
    value = metric.calculate(actual, predicted)
    assert isinstance(value, float)
    assert value >= 0

def test_invalid_metric():
    with pytest.raises(ValueError):
        MetricFactory.create_metric('invalid_metric')

# Add more test cases...