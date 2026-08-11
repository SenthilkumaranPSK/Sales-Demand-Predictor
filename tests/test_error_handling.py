"""
Unit tests for API error handling and HTTP status code mapping.

Tests cover:
- Custom exception classes
- Error handler middleware
- HTTP status code mapping
- Structured JSON error responses
- Error logging

**Validates: Requirements 5.4, 5.7, 10.2**
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from datetime import datetime
from unittest.mock import Mock, patch
import json

from src.api.exceptions import (
    APIException,
    ValidationError,
    ResourceNotFoundError,
    ModelNotFoundError,
    InternalServerError,
    ServiceUnavailableError,
    ForecastGenerationError,
    AuthenticationError,
    RateLimitError
)
from src.api.error_handlers import (
    api_exception_handler,
    validation_exception_handler,
    value_error_handler,
    runtime_error_handler,
    connection_error_handler,
    global_exception_handler,
    register_error_handlers
)


class TestCustomExceptions:
    """Test suite for custom exception classes."""
    
    def test_validation_error_attributes(self):
        """Test ValidationError has correct status code and error code."""
        exc = ValidationError(
            message="Invalid forecast horizon",
            details={"field": "forecast_horizon", "value": -1}
        )
        
        assert exc.status_code == 400
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.message == "Invalid forecast horizon"
        assert exc.details == {"field": "forecast_horizon", "value": -1}
    
    def test_model_not_found_error_attributes(self):
        """Test ModelNotFoundError has correct status code and error code."""
        exc = ModelNotFoundError(
            message="Model not found",
            details={"model_id": "test_model"}
        )
        
        assert exc.status_code == 404
        assert exc.error_code == "MODEL_NOT_FOUND"
        assert exc.message == "Model not found"
    
    def test_resource_not_found_error_attributes(self):
        """Test ResourceNotFoundError has correct status code and error code."""
        exc = ResourceNotFoundError(
            message="Resource not found",
            details={"resource_type": "product"}
        )
        
        assert exc.status_code == 404
        assert exc.error_code == "RESOURCE_NOT_FOUND"
    
    def test_internal_server_error_attributes(self):
        """Test InternalServerError has correct status code and error code."""
        exc = InternalServerError(
            message="Internal error occurred",
            details={"error": "Database connection failed"}
        )
        
        assert exc.status_code == 500
        assert exc.error_code == "INTERNAL_ERROR"
    
    def test_service_unavailable_error_attributes(self):
        """Test ServiceUnavailableError has correct status code and error code."""
        exc = ServiceUnavailableError(
            message="Service unavailable",
            details={"service": "forecasting_engine"}
        )
        
        assert exc.status_code == 503
        assert exc.error_code == "SERVICE_UNAVAILABLE"
    
    def test_forecast_generation_error_attributes(self):
        """Test ForecastGenerationError has correct status code and error code."""
        exc = ForecastGenerationError(
            message="Forecast generation failed",
            details={"model_id": "test_model"}
        )
        
        assert exc.status_code == 500
        assert exc.error_code == "FORECAST_ERROR"
    
    def test_authentication_error_attributes(self):
        """Test AuthenticationError has correct status code and error code."""
        exc = AuthenticationError(
            message="Invalid API key",
            details={"reason": "expired"}
        )
        
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_ERROR"
    
    def test_rate_limit_error_attributes(self):
        """Test RateLimitError has correct status code and error code."""
        exc = RateLimitError(
            message="Rate limit exceeded",
            details={"limit": 1000, "window": "1 minute"}
        )
        
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
    
    def test_custom_error_code_override(self):
        """Test that error_code can be overridden in constructor."""
        exc = APIException(
            message="Custom error",
            error_code="CUSTOM_CODE"
        )
        
        assert exc.error_code == "CUSTOM_CODE"
    
    def test_exception_without_details(self):
        """Test exception can be created without details."""
        exc = ValidationError(message="Simple error")
        
        assert exc.details == {}


class TestErrorHandlers:
    """Test suite for error handler functions."""
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = Mock(spec=Request)
        request.url.path = "/api/v1/forecast"
        request.method = "POST"
        return request
    
    @pytest.mark.asyncio
    async def test_api_exception_handler_returns_correct_status(self, mock_request):
        """Test api_exception_handler returns correct HTTP status code."""
        exc = ValidationError(
            message="Invalid parameter",
            details={"field": "forecast_horizon"}
        )
        
        response = await api_exception_handler(mock_request, exc)
        
        assert response.status_code == 400
        
        # Parse response body
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Invalid parameter"
        assert "timestamp" in body["error"]
        assert body["error"]["details"] == {"field": "forecast_horizon"}
    
    @pytest.mark.asyncio
    async def test_api_exception_handler_without_details(self, mock_request):
        """Test api_exception_handler works without details."""
        exc = ModelNotFoundError(message="Model not found")
        
        response = await api_exception_handler(mock_request, exc)
        
        assert response.status_code == 404
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "MODEL_NOT_FOUND"
        assert "details" not in body["error"]
    
    @pytest.mark.asyncio
    async def test_value_error_handler_maps_to_400(self, mock_request):
        """Test value_error_handler maps ValueError to HTTP 400."""
        exc = ValueError("Invalid forecast horizon: must be between 1 and 90")
        
        response = await value_error_handler(mock_request, exc)
        
        assert response.status_code == 400
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "Invalid forecast horizon" in body["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_runtime_error_handler_maps_to_500(self, mock_request):
        """Test runtime_error_handler maps RuntimeError to HTTP 500."""
        exc = RuntimeError("Model prediction failed")
        
        response = await runtime_error_handler(mock_request, exc)
        
        assert response.status_code == 500
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "internal error occurred" in body["error"]["message"]
        assert body["error"]["details"]["error"] == "Model prediction failed"
    
    @pytest.mark.asyncio
    async def test_connection_error_handler_maps_to_503(self, mock_request):
        """Test connection_error_handler maps ConnectionError to HTTP 503."""
        exc = ConnectionError("Database connection failed")
        
        response = await connection_error_handler(mock_request, exc)
        
        assert response.status_code == 503
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert "service is currently unavailable" in body["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_timeout_error_handler_maps_to_503(self, mock_request):
        """Test connection_error_handler maps TimeoutError to HTTP 503."""
        exc = TimeoutError("Request timeout")
        
        response = await connection_error_handler(mock_request, exc)
        
        assert response.status_code == 503
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
    
    @pytest.mark.asyncio
    async def test_global_exception_handler_maps_to_500(self, mock_request):
        """Test global_exception_handler maps any Exception to HTTP 500."""
        exc = Exception("Unexpected error")
        
        response = await global_exception_handler(mock_request, exc)
        
        assert response.status_code == 500
        
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["message"] == "An internal error occurred"
    
    @pytest.mark.asyncio
    async def test_error_response_includes_timestamp(self, mock_request):
        """Test all error responses include timestamp."""
        exc = ValidationError(message="Test error")
        
        response = await api_exception_handler(mock_request, exc)
        body = json.loads(response.body.decode())
        
        assert "timestamp" in body["error"]
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(body["error"]["timestamp"])


class TestErrorHandlerRegistration:
    """Test suite for error handler registration."""
    
    def test_register_error_handlers_adds_all_handlers(self):
        """Test register_error_handlers adds all exception handlers."""
        app = FastAPI()
        
        # Count handlers before registration
        initial_handler_count = len(app.exception_handlers)
        
        register_error_handlers(app)
        
        # Verify handlers were added
        assert len(app.exception_handlers) > initial_handler_count
        
        # Verify specific handlers are registered
        assert APIException in app.exception_handlers
        assert ValueError in app.exception_handlers
        assert RuntimeError in app.exception_handlers
        assert ConnectionError in app.exception_handlers
        assert TimeoutError in app.exception_handlers
        assert Exception in app.exception_handlers


class TestErrorHandlingIntegration:
    """Integration tests for error handling in FastAPI application."""
    
    @pytest.fixture
    def app(self):
        """Create a test FastAPI application with error handlers."""
        # Disable default server error middleware to test our handlers
        from starlette.middleware.errors import ServerErrorMiddleware
        
        app = FastAPI()
        
        # Remove default error middleware
        app.user_middleware = []
        
        register_error_handlers(app)
        
        # Add test endpoints
        @app.get("/test/validation-error")
        async def test_validation_error():
            raise ValidationError(
                message="Test validation error",
                details={"field": "test_field"}
            )
        
        @app.get("/test/model-not-found")
        async def test_model_not_found():
            raise ModelNotFoundError(
                message="Model not found",
                details={"model_id": "test_model"}
            )
        
        @app.get("/test/internal-error")
        async def test_internal_error():
            raise InternalServerError(message="Test internal error")
        
        @app.get("/test/service-unavailable")
        async def test_service_unavailable():
            raise ServiceUnavailableError(message="Service unavailable")
        
        @app.get("/test/value-error")
        async def test_value_error():
            raise ValueError("Test value error")
        
        @app.get("/test/runtime-error")
        async def test_runtime_error():
            raise RuntimeError("Test runtime error")
        
        @app.get("/test/connection-error")
        async def test_connection_error():
            raise ConnectionError("Test connection error")
        
        @app.get("/test/unexpected-error")
        async def test_unexpected_error():
            raise Exception("Unexpected error")
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        # Don't raise server exceptions - let them be handled by our error handlers
        return TestClient(app, raise_server_exceptions=False)
    
    def test_validation_error_returns_400(self, client):
        """Test ValidationError returns HTTP 400."""
        response = client.get("/test/validation-error")
        
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    
    def test_model_not_found_returns_404(self, client):
        """Test ModelNotFoundError returns HTTP 404."""
        response = client.get("/test/model-not-found")
        
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    
    def test_internal_error_returns_500(self, client):
        """Test InternalServerError returns HTTP 500."""
        response = client.get("/test/internal-error")
        
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    
    def test_service_unavailable_returns_503(self, client):
        """Test ServiceUnavailableError returns HTTP 503."""
        response = client.get("/test/service-unavailable")
        
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    
    def test_value_error_returns_400(self, client):
        """Test ValueError returns HTTP 400."""
        response = client.get("/test/value-error")
        
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    
    def test_runtime_error_returns_500(self, client):
        """Test RuntimeError returns HTTP 500."""
        response = client.get("/test/runtime-error")
        
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    
    def test_connection_error_returns_503(self, client):
        """Test ConnectionError returns HTTP 503."""
        response = client.get("/test/connection-error")
        
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    
    def test_unexpected_error_returns_500(self, client):
        """Test unexpected Exception returns HTTP 500."""
        response = client.get("/test/unexpected-error")
        
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    
    def test_error_response_structure(self, client):
        """Test error responses have correct structure."""
        response = client.get("/test/validation-error")
        
        error = response.json()["error"]
        
        # Verify required fields
        assert "code" in error
        assert "message" in error
        assert "timestamp" in error
        
        # Verify optional fields
        assert "details" in error
    
    def test_error_response_content_type(self, client):
        """Test error responses have JSON content type."""
        response = client.get("/test/validation-error")
        
        assert "application/json" in response.headers["content-type"]


class TestHTTPStatusCodeMapping:
    """Test suite for HTTP status code mapping.
    
    **Validates: Requirement 10.2 - Property 11: HTTP Status Code Mapping**
    """
    
    def test_validation_errors_map_to_400(self):
        """Test validation errors map to HTTP 400."""
        exc = ValidationError(message="Validation failed")
        assert exc.status_code == 400
    
    def test_missing_resources_map_to_404(self):
        """Test missing resources map to HTTP 404."""
        exc1 = ResourceNotFoundError(message="Resource not found")
        exc2 = ModelNotFoundError(message="Model not found")
        
        assert exc1.status_code == 404
        assert exc2.status_code == 404
    
    def test_internal_errors_map_to_500(self):
        """Test internal errors map to HTTP 500."""
        exc1 = InternalServerError(message="Internal error")
        exc2 = ForecastGenerationError(message="Forecast failed")
        
        assert exc1.status_code == 500
        assert exc2.status_code == 500
    
    def test_service_unavailability_maps_to_503(self):
        """Test service unavailability maps to HTTP 503."""
        exc = ServiceUnavailableError(message="Service unavailable")
        assert exc.status_code == 503
    
    def test_authentication_errors_map_to_401(self):
        """Test authentication errors map to HTTP 401."""
        exc = AuthenticationError(message="Authentication failed")
        assert exc.status_code == 401
    
    def test_rate_limit_errors_map_to_429(self):
        """Test rate limit errors map to HTTP 429."""
        exc = RateLimitError(message="Rate limit exceeded")
        assert exc.status_code == 429
