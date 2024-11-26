# transformer_conv_attention/data/generators/synthetic_generator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from .base_generator import BaseDataGenerator
from ...utils.logger import get_logger

logger = get_logger(__name__)

class SyntheticTimeSeriesGenerator(BaseDataGenerator):
    """Generator for synthetic time series data (Strategy Pattern)"""

    def __init__(
            self,
            n_samples: int = 1000,
            output_path: Optional[str] = None,
            noise_level: float = 0.1
    ):
        super().__init__(output_path)
        self.n_samples = n_samples
        self.noise_level = noise_level
        logger.info(f"[DEBUG] Initialized SyntheticTimeSeriesGenerator with {n_samples} samples")

    def generate(self) -> pd.DataFrame:
        """Generate synthetic time series data"""
        logger.info("[DEBUG] Starting synthetic data generation")

        # Create timestamps
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(self.n_samples)]

        # Generate synthetic patterns
        t = np.arange(self.n_samples)

        # Create target variable (combination of patterns)
        target = (
                10 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
                + 5 * np.sin(2 * np.pi * t / 24)       # Daily pattern
                + np.random.normal(0, self.noise_level, self.n_samples)
        )

        # Create features
        feature1 = (
                8 * np.sin(2 * np.pi * t / (24 * 7) + 1)
                + np.random.normal(0, self.noise_level/2, self.n_samples)
        )

        feature2 = (
                4 * np.sin(2 * np.pi * t / 24 + 2)
                + np.random.normal(0, self.noise_level/2, self.n_samples)
        )

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': target,
            'feature1': feature1,
            'feature2': feature2
        })

        logger.info(f"[DEBUG] Generated DataFrame with shape: {df.shape}")
        logger.info(f"[DEBUG] Columns: {df.columns.tolist()}")
        logger.info(f"[DEBUG] Value range: ({target.min():.2f}, {target.max():.2f})")

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Additional validation for synthetic data"""
        # Call parent validation
        super().validate(df)

        # Specific validations for synthetic data
        assert 'value' in df.columns, "Missing target column 'value'"
        assert 'feature1' in df.columns, "Missing feature1 column"
        assert 'feature2' in df.columns, "Missing feature2 column"

        # Check for NaN values
        assert not df.isnull().any().any(), "Generated data contains NaN values"

        return True