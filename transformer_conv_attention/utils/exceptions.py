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

class BaseError(Exception):
    """Base exception class for the project"""
    pass

class StageError(BaseError):
    """Exception raised for pipeline stage errors"""
    pass

class PipelineError(BaseError):
    """Exception raised for pipeline-related errors"""
    pass

class ValidationError(BaseError):
    """Exception raised for validation errors"""
    pass

class ProcessingError(BaseError):
    """Exception raised for data processing errors"""
    pass

class TrainingError(BaseError):
    """Exception raised for training-related errors"""
    pass

class EvaluationError(BaseError):
    """Exception raised for evaluation-related errors"""
    pass
