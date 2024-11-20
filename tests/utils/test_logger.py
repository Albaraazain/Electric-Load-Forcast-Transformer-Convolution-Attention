# tests/utils/test_logger.py
import pytest
from pathlib import Path
from transformer_conv_attention.utils import get_logger, setup_logger

def test_logger_creation(tmp_path):
    # Test basic logger
    logger = get_logger("test")
    assert logger.name == "test"

    # Test logger with file
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_file", str(log_file))
    assert log_file.exists()
    logger.info("Test message")
    assert log_file.read_text().strip().endswith("Test message")