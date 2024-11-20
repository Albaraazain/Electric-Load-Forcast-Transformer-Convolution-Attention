# transformer_conv_attention/utils/exceptions.py
class BaseError(Exception):
    """Base exception for all custom errors"""
    pass

class ConfigurationError(BaseError):
    """Raised when configuration validation fails"""
    pass

class ModelError(BaseError):
    """Raised when model operations fail"""
    pass

class DataError(BaseError):
    """Raised when data operations fail"""
    pass