"""
Custom exception classes for the Demand Forecasting System API.

This module defines custom exceptions that map to specific HTTP status codes
and provide structured error responses.

**Validates: Requirements 5.4, 5.7, 10.2**
"""

from typing import Optional, Dict, Any


class APIException(Exception):
    """
    Base exception class for API errors.
    
    All custom API exceptions inherit from this class and define:
    - status_code: HTTP status code to return
    - error_code: Machine-readable error code
    - message: Human-readable error message
    - details: Optional additional error details
    """
    
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize API exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
            error_code: Optional override for the error code
        """
        self.message = message
        self.details = details or {}
        if error_code:
            self.error_code = error_code
        super().__init__(self.message)


class ValidationError(APIException):
    """
    Exception for validation errors (HTTP 400).
    
    Raised when request parameters fail validation:
    - Invalid forecast horizon
    - Missing required fields
    - Invalid data types
    - Out-of-range values
    
    **Validates: Requirement 5.4**
    """
    
    status_code = 400
    error_code = "VALIDATION_ERROR"


class ResourceNotFoundError(APIException):
    """
    Exception for missing resources (HTTP 404).
    
    Raised when requested resources don't exist:
    - Model not found in registry
    - Product not found
    - Dataset not found
    
    **Validates: Requirement 10.2**
    """
    
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"


class ModelNotFoundError(ResourceNotFoundError):
    """
    Exception for missing models (HTTP 404).
    
    Raised when a requested model doesn't exist in the Model Registry.
    
    **Validates: Requirement 5.4**
    """
    
    error_code = "MODEL_NOT_FOUND"


class InternalServerError(APIException):
    """
    Exception for internal server errors (HTTP 500).
    
    Raised when unexpected errors occur:
    - Model loading failures
    - Prediction failures
    - Database errors
    - Unexpected exceptions
    
    **Validates: Requirement 10.2**
    """
    
    status_code = 500
    error_code = "INTERNAL_ERROR"


class ServiceUnavailableError(APIException):
    """
    Exception for service unavailability (HTTP 503).
    
    Raised when required services are unavailable:
    - Forecasting engine unavailable
    - Database connection failed
    - External service timeout
    
    **Validates: Requirements 5.7, 10.2**
    """
    
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"


class ForecastGenerationError(InternalServerError):
    """
    Exception for forecast generation failures (HTTP 500).
    
    Raised when forecast generation fails due to:
    - Model prediction errors
    - Feature processing errors
    - Confidence interval calculation errors
    
    **Validates: Requirement 5.4**
    """
    
    error_code = "FORECAST_ERROR"


class AuthenticationError(APIException):
    """
    Exception for authentication failures (HTTP 401).
    
    Raised when API key authentication fails:
    - Missing API key
    - Invalid API key
    - Expired API key
    
    **Validates: Requirement 5.5**
    """
    
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"


class RateLimitError(APIException):
    """
    Exception for rate limit exceeded (HTTP 429).
    
    Raised when request rate exceeds limits:
    - Too many requests per minute
    - Concurrent request limit exceeded
    
    **Validates: Requirement 7.3**
    """
    
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
