"""
Error handler middleware for the Demand Forecasting System API.

This module provides centralized error handling that maps exceptions to
appropriate HTTP status codes and returns structured JSON error responses.

**Validates: Requirements 5.4, 5.7, 10.2**
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime, timezone
from typing import Union
import traceback

from src.api.exceptions import APIException
from src.utils.logging_config import logger


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """
    Handle custom API exceptions.
    
    Maps APIException subclasses to appropriate HTTP status codes and
    returns structured JSON error responses.
    
    **Validates: Requirements 5.4, 10.2**
    
    Args:
        request: FastAPI request object
        exc: APIException instance
        
    Returns:
        JSONResponse with error details and appropriate status code
    """
    # Log the error
    logger.error(
        f"API exception: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    # Build error response
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    # Add details if present
    if exc.details:
        error_response["error"]["details"] = exc.details
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle FastAPI/Pydantic validation errors.
    
    Maps request validation errors to HTTP 400 with structured error response.
    
    **Validates: Requirement 5.4**
    
    Args:
        request: FastAPI request object
        exc: RequestValidationError from Pydantic
        
    Returns:
        JSONResponse with validation error details and HTTP 400 status
    """
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        validation_errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    # Log the validation error
    logger.warning(
        f"Validation error on {request.url.path}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "validation_errors": validation_errors
        }
    )
    
    # Build error response
    error_response = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "validation_errors": validation_errors
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Handle ValueError exceptions.
    
    Maps ValueError to HTTP 400 (validation error) with structured response.
    
    **Validates: Requirement 5.4**
    
    Args:
        request: FastAPI request object
        exc: ValueError instance
        
    Returns:
        JSONResponse with error details and HTTP 400 status
    """
    # Log the error
    logger.warning(
        f"ValueError on {request.url.path}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc)
        }
    )
    
    # Build error response
    error_response = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response
    )


async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """
    Handle RuntimeError exceptions.
    
    Maps RuntimeError to HTTP 500 (internal error) with structured response.
    
    **Validates: Requirement 10.2**
    
    Args:
        request: FastAPI request object
        exc: RuntimeError instance
        
    Returns:
        JSONResponse with error details and HTTP 500 status
    """
    # Log the error with stack trace
    logger.error(
        f"RuntimeError on {request.url.path}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc)
        },
        exc_info=True
    )
    
    # Build error response
    error_response = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal error occurred while processing the request",
            "details": {
                "error": str(exc)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


async def connection_error_handler(
    request: Request,
    exc: Union[ConnectionError, TimeoutError]
) -> JSONResponse:
    """
    Handle connection and timeout errors.
    
    Maps connection/timeout errors to HTTP 503 (service unavailable).
    
    **Validates: Requirements 5.7, 10.2**
    
    Args:
        request: FastAPI request object
        exc: ConnectionError or TimeoutError instance
        
    Returns:
        JSONResponse with error details and HTTP 503 status
    """
    # Log the error
    logger.error(
        f"Connection error on {request.url.path}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__
        }
    )
    
    # Build error response
    error_response = {
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "A required service is currently unavailable. Please try again later.",
            "details": {
                "error_type": type(exc).__name__
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_response
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions.
    
    Catches any exceptions not handled by specific handlers and returns
    HTTP 500 with structured error response.
    
    **Validates: Requirement 10.2**
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse with error details and HTTP 500 status
    """
    # Log the error with full stack trace
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__
        },
        exc_info=True
    )
    
    # Build error response (don't expose internal details in production)
    error_response = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


def register_error_handlers(app):
    """
    Register all error handlers with the FastAPI application.
    
    This function should be called during application initialization to
    register all custom exception handlers.
    
    **Validates: Requirements 5.4, 5.7, 10.2**
    
    Args:
        app: FastAPI application instance
    """
    # Register custom API exception handler
    app.add_exception_handler(APIException, api_exception_handler)
    
    # Register Pydantic validation error handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Register standard Python exception handlers
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(RuntimeError, runtime_error_handler)
    app.add_exception_handler(ConnectionError, connection_error_handler)
    app.add_exception_handler(TimeoutError, connection_error_handler)
    
    # Register global exception handler (catch-all)
    app.add_exception_handler(Exception, global_exception_handler)
    
    logger.info("Error handlers registered successfully")
