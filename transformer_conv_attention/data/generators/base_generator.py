# transformer_conv_attention/data/generators/base_generator.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class BaseDataGenerator(ABC):
    """Abstract base class for data generators (Template Method Pattern)"""

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path

    def generate_and_save(self) -> pd.DataFrame:
        """Template method defining the generation workflow"""
        print("\n[DEBUG] Starting data generation process...")

        # Generate data
        df = self.generate()
        print(f"[DEBUG] Generated data shape: {df.shape}")

        # Validate data
        self.validate(df)
        print("[DEBUG] ✓ Data validation passed")

        # Save if path provided
        if self.output_path:
            self.save(df)
            print(f"[DEBUG] ✓ Saved data to {self.output_path}")

        return df

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generate the data"""
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate generated data"""
        print("[DEBUG] Running base validation checks...")

        # Basic validation
        assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"
        print("[DEBUG] ✓ Output is a pandas DataFrame")

        assert not df.empty, "Generated data is empty"
        print("[DEBUG] ✓ DataFrame is not empty")

        assert 'timestamp' in df.columns, "Missing timestamp column"
        print("[DEBUG] ✓ Timestamp column present")

        return True

    def save(self, df: pd.DataFrame) -> None:
        """Save data to file"""
        if self.output_path:
            print(f"[DEBUG] Saving data to {self.output_path}")
            df.to_csv(self.output_path, index=False)