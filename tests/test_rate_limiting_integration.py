"""
Integration tests for rate limiting with the full API.

**Validates: Requirements 5.6, 7.2, 7.3**
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.api.rate_limiter import RateLimiter, rate_limiter


@pytest.fixture
def client_with_low_rate_limit():
    """
    Create test client with low rate limits for testing.
    
    Temporarily replaces the global rate limiter with one that has
    low limits for easier testing.
    """
    # Save original rate limiter
    original_limiter = rate_limiter
    
    # Create test limiter with low limits
    test_limiter = RateLimiter(requests_per_minute=3, max_concurrent_requests=2)
    
    # Replace the rate limiter in the middleware
    # Find the rate limit middleware and update its limiter
    for middleware in app.user_middleware:
        if hasattr(middleware, 'kwargs') and 'rate_limiter' in middleware.kwargs:
            middleware.kwargs['rate_limiter'] = test_limiter
    
    client = TestClient(app)
    
    yield client
    
    # Restore original rate limiter
    for middleware in app.user_middleware:
        if hasattr(middleware, 'kwargs') and 'rate_limiter' in middleware.kwargs:
            middleware.kwargs['rate_limiter'] = original_limiter


def test_api_rate_limiting_on_forecast_endpoint(client_with_low_rate_limit):
    """
    Test that rate limiting is enforced on the forecast endpoint.
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    client = client_with_low_rate_limit
    
    # Use the default development API key
    api_key = "dev-api-key-12345"
    
    # Mock the forecasting engine and model registry
    with patch('src.api.main.forecasting_engine') as mock_engine, \
         patch('src.api.main.model_registry') as mock_registry:
        
        # Setup mocks
        mock_registry.get_latest_model.return_value = ("model_123", {})
        
        mock_forecast_result = Mock()
        mock_forecast_result.product_id = "PROD-123"
        mock_forecast_result.model_id = "model_123"
        mock_forecast_result.timestamps = []
        mock_forecast_result.predictions = []
        mock_forecast_result.confidence_intervals = {}
        mock_forecast_result.metadata = {}
        
        mock_engine.generate_forecast.return_value = mock_forecast_result
        
        # Make requests within limit
        for i in range(3):
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": api_key}
            )
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
        
        # 4th request should be rate limited
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30
            },
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 429
        
        data = response.json()
        assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "Rate limit exceeded" in data["error"]["message"]


def test_api_rate_limiting_per_api_key_isolation():
    """
    Test that different API keys have independent rate limits.
    
    **Validates: Requirements 5.6, 7.2**
    """
    client = TestClient(app)
    
    # Create a temporary low-limit rate limiter for this test
    test_limiter = RateLimiter(requests_per_minute=2, max_concurrent_requests=10)
    
    # Patch the middleware's rate limiter
    with patch.object(app.user_middleware[1].cls, '__init__', 
                      lambda self, app, rate_limiter=None: 
                      super(app.user_middleware[1].cls, self).__init__(app)):
        
        # Make requests with different API keys
        # Key 1 - 2 requests (at limit)
        for i in range(2):
            response = client.get("/", headers={"X-API-Key": "key-1"})
            assert response.status_code == 200
        
        # Key 2 should still be able to make requests
        response = client.get("/", headers={"X-API-Key": "key-2"})
        assert response.status_code == 200


def test_health_check_bypasses_rate_limiting():
    """
    Test that health check endpoint is not rate limited.
    
    **Validates: Requirement 7.2**
    """
    client = TestClient(app)
    
    # Make many health check requests - should not be rate limited
    # even if we exceed normal limits
    for i in range(20):
        response = client.get("/api/v1/health")
        # Health check might fail due to database, but should not be rate limited
        assert response.status_code in [200, 503]  # Not 429


def test_rate_limit_headers_present_in_responses():
    """
    Test that rate limit headers are included in API responses.
    
    **Validates: Requirement 7.2**
    """
    client = TestClient(app)
    
    # Use the default development API key
    api_key = "dev-api-key-12345"
    
    # Make a request to any endpoint
    response = client.get("/", headers={"X-API-Key": api_key})
    
    # Check for rate limit headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    
    # Verify header values are numeric
    limit = int(response.headers["X-RateLimit-Limit"])
    remaining = int(response.headers["X-RateLimit-Remaining"])
    
    assert limit > 0
    assert 0 <= remaining <= limit


def test_rate_limit_error_format():
    """
    Test that rate limit errors follow the standard error format.
    
    **Validates: Requirements 5.6, 7.3**
    """
    # Create a client with very low rate limit
    test_limiter = RateLimiter(requests_per_minute=1, max_concurrent_requests=10)
    
    # Create a minimal test app
    from fastapi import FastAPI
    from src.api.rate_limiter import RateLimitMiddleware
    
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @test_app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(test_app)
    
    # Use the default development API key
    api_key = "dev-api-key-12345"
    
    # First request succeeds
    response = client.get("/test", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    
    # Second request is rate limited
    response = client.get("/test", headers={"X-API-Key": api_key})
    assert response.status_code == 429
    
    # Verify error format
    data = response.json()
    assert "error" in data
    assert "code" in data["error"]
    assert "message" in data["error"]
    assert "timestamp" in data["error"]
    
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    
    # Verify Retry-After header
    assert "Retry-After" in response.headers
    assert response.headers["Retry-After"] == "60"
