"""
Tests for rate limiting middleware.

**Validates: Requirements 5.6, 7.2, 7.3**
"""

import pytest
import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.rate_limiter import RateLimiter, RateLimitMiddleware
from config.settings import settings


# ============================================================================
# Unit Tests for RateLimiter
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_within_limit():
    """
    Test that rate limiter allows requests within the per-minute limit.
    
    **Validates: Requirement 7.2**
    """
    limiter = RateLimiter(requests_per_minute=10, max_concurrent_requests=5)
    
    # Create mock request
    request = Mock(spec=Request)
    request.headers.get.return_value = "test-api-key"
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Make 10 requests (within limit)
    for i in range(10):
        is_allowed, error_msg = await limiter.check_rate_limit(request)
        assert is_allowed is True
        assert error_msg is None
        
        # Release concurrent slot
        await limiter.release_concurrent_slot(request)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_exceeding_per_minute_limit():
    """
    Test that rate limiter blocks requests exceeding per-minute limit.
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    limiter = RateLimiter(requests_per_minute=5, max_concurrent_requests=10)
    
    # Create mock request
    request = Mock(spec=Request)
    request.headers.get.return_value = "test-api-key"
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Make 5 requests (at limit)
    for i in range(5):
        is_allowed, error_msg = await limiter.check_rate_limit(request)
        assert is_allowed is True
        await limiter.release_concurrent_slot(request)
    
    # 6th request should be blocked
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is False
    assert error_msg is not None
    assert "Rate limit exceeded" in error_msg
    assert "5 requests per minute" in error_msg


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_exceeding_concurrent_limit():
    """
    Test that rate limiter blocks requests exceeding concurrent request limit.
    
    **Validates: Requirements 5.6, 7.3**
    """
    limiter = RateLimiter(requests_per_minute=100, max_concurrent_requests=3)
    
    # Create mock request
    request = Mock(spec=Request)
    request.headers.get.return_value = "test-api-key"
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Make 3 concurrent requests (at limit)
    for i in range(3):
        is_allowed, error_msg = await limiter.check_rate_limit(request)
        assert is_allowed is True
        # Don't release slots yet - simulating concurrent requests
    
    # 4th concurrent request should be blocked
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is False
    assert error_msg is not None
    assert "Concurrent request limit exceeded" in error_msg
    assert "3 concurrent requests" in error_msg
    
    # Release one slot
    await limiter.release_concurrent_slot(request)
    
    # Now another request should be allowed
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_sliding_window_cleanup():
    """
    Test that rate limiter cleans up old timestamps (sliding window).
    
    **Validates: Requirement 7.2**
    """
    limiter = RateLimiter(requests_per_minute=5, max_concurrent_requests=10)
    
    # Create mock request
    request = Mock(spec=Request)
    request.headers.get.return_value = "test-api-key"
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Make 5 requests (at limit)
    for i in range(5):
        is_allowed, error_msg = await limiter.check_rate_limit(request)
        assert is_allowed is True
        await limiter.release_concurrent_slot(request)
    
    # 6th request should be blocked
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is False
    
    # Wait for 61 seconds (simulate time passing)
    # Mock time to avoid actual waiting
    current_time = time.time()
    with patch('src.api.rate_limiter.time.time') as mock_time:
        # Set time to 61 seconds in the future
        mock_time.return_value = current_time + 61
        
        # Now request should be allowed (old timestamps cleaned up)
        is_allowed, error_msg = await limiter.check_rate_limit(request)
        assert is_allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_different_clients_independent():
    """
    Test that rate limits are tracked independently per client.
    
    **Validates: Requirement 7.2**
    """
    limiter = RateLimiter(requests_per_minute=3, max_concurrent_requests=2)
    
    # Create two different clients
    request1 = Mock(spec=Request)
    request1.headers.get.return_value = "api-key-1"
    request1.client = Mock()
    request1.client.host = "127.0.0.1"
    
    request2 = Mock(spec=Request)
    request2.headers.get.return_value = "api-key-2"
    request2.client = Mock()
    request2.client.host = "192.168.1.1"
    
    # Client 1 makes 3 requests (at limit)
    for i in range(3):
        is_allowed, error_msg = await limiter.check_rate_limit(request1)
        assert is_allowed is True
        await limiter.release_concurrent_slot(request1)
    
    # Client 1's 4th request should be blocked
    is_allowed, error_msg = await limiter.check_rate_limit(request1)
    assert is_allowed is False
    
    # Client 2 should still be able to make requests
    for i in range(3):
        is_allowed, error_msg = await limiter.check_rate_limit(request2)
        assert is_allowed is True
        await limiter.release_concurrent_slot(request2)


@pytest.mark.asyncio
async def test_rate_limiter_uses_api_key_over_ip():
    """
    Test that rate limiter prefers API key over IP address for identification.
    
    **Validates: Requirement 5.6**
    """
    limiter = RateLimiter(requests_per_minute=2, max_concurrent_requests=5)
    
    # Request with API key
    request_with_key = Mock(spec=Request)
    request_with_key.headers.get.return_value = "test-api-key"
    request_with_key.client = Mock()
    request_with_key.client.host = "127.0.0.1"
    
    # Request without API key (same IP)
    request_without_key = Mock(spec=Request)
    request_without_key.headers.get.return_value = None
    request_without_key.client = Mock()
    request_without_key.client.host = "127.0.0.1"
    
    # Make 2 requests with API key (at limit)
    for i in range(2):
        is_allowed, error_msg = await limiter.check_rate_limit(request_with_key)
        assert is_allowed is True
        await limiter.release_concurrent_slot(request_with_key)
    
    # 3rd request with API key should be blocked
    is_allowed, error_msg = await limiter.check_rate_limit(request_with_key)
    assert is_allowed is False
    
    # Request without API key (using IP) should still be allowed
    # because it's tracked separately
    is_allowed, error_msg = await limiter.check_rate_limit(request_without_key)
    assert is_allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_handles_x_forwarded_for():
    """
    Test that rate limiter correctly extracts IP from X-Forwarded-For header.
    
    **Validates: Requirement 5.6**
    """
    limiter = RateLimiter(requests_per_minute=10, max_concurrent_requests=5)
    
    # Create mock request with X-Forwarded-For header
    request = Mock(spec=Request)
    
    def mock_header_get(header_name):
        if header_name == settings.api_key_header:
            return None
        elif header_name == "X-Forwarded-For":
            return "203.0.113.1, 198.51.100.1"  # Multiple IPs in chain
        return None
    
    request.headers.get = mock_header_get
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Check that it uses the first IP from X-Forwarded-For
    client_id = limiter._get_client_identifier(request)
    assert client_id == "ip:203.0.113.1"


def test_rate_limiter_get_stats():
    """
    Test rate limiter statistics retrieval.
    
    **Validates: Requirement 7.2**
    """
    limiter = RateLimiter(requests_per_minute=100, max_concurrent_requests=10)
    
    # Get global stats
    stats = limiter.get_stats()
    assert "total_clients" in stats
    assert "requests_per_minute_limit" in stats
    assert stats["requests_per_minute_limit"] == 100
    assert stats["max_concurrent_requests"] == 10


# ============================================================================
# Integration Tests with FastAPI
# ============================================================================


def test_rate_limit_middleware_integration():
    """
    Test rate limiting middleware integration with FastAPI.
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    # Create test app
    app = FastAPI()
    
    # Add rate limiting middleware with low limits for testing
    test_limiter = RateLimiter(requests_per_minute=5, max_concurrent_requests=10)
    app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # Make 5 requests (within limit)
    for i in range(5):
        response = client.get("/test", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    # 6th request should be rate limited
    response = client.get("/test", headers={"X-API-Key": "test-key"})
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    
    data = response.json()
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Rate limit exceeded" in data["error"]["message"]


def test_rate_limit_middleware_skips_health_check():
    """
    Test that rate limiting middleware skips health check endpoint.
    
    **Validates: Requirement 7.2**
    """
    # Create test app
    app = FastAPI()
    
    # Add rate limiting middleware with very low limit
    test_limiter = RateLimiter(requests_per_minute=1, max_concurrent_requests=1)
    app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "healthy"}
    
    client = TestClient(app)
    
    # Make multiple health check requests - should not be rate limited
    for i in range(10):
        response = client.get("/api/v1/health")
        assert response.status_code == 200


def test_rate_limit_middleware_returns_429_with_proper_format():
    """
    Test that rate limit errors return HTTP 429 with proper error format.
    
    **Validates: Requirements 5.6, 7.3**
    """
    # Create test app
    app = FastAPI()
    
    # Add rate limiting middleware with limit of 1
    test_limiter = RateLimiter(requests_per_minute=1, max_concurrent_requests=10)
    app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # First request succeeds
    response = client.get("/test", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    
    # Second request is rate limited
    response = client.get("/test", headers={"X-API-Key": "test-key"})
    assert response.status_code == 429
    
    # Check error response format
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "message" in data["error"]
    assert "timestamp" in data["error"]
    
    # Check headers
    assert response.headers["Retry-After"] == "60"
    assert "X-RateLimit-Limit" in response.headers
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limit_per_api_key():
    """
    Test that rate limits are enforced per API key.
    
    **Validates: Requirements 5.6, 7.2**
    """
    # Create test app
    app = FastAPI()
    
    # Add rate limiting middleware
    test_limiter = RateLimiter(requests_per_minute=2, max_concurrent_requests=10)
    app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # API key 1 makes 2 requests (at limit)
    for i in range(2):
        response = client.get("/test", headers={"X-API-Key": "key-1"})
        assert response.status_code == 200
    
    # API key 1's 3rd request is blocked
    response = client.get("/test", headers={"X-API-Key": "key-1"})
    assert response.status_code == 429
    
    # API key 2 can still make requests
    response = client.get("/test", headers={"X-API-Key": "key-2"})
    assert response.status_code == 200


def test_rate_limit_headers_in_response():
    """
    Test that rate limit information is included in response headers.
    
    **Validates: Requirement 7.2**
    """
    # Create test app
    app = FastAPI()
    
    # Add rate limiting middleware
    test_limiter = RateLimiter(requests_per_minute=10, max_concurrent_requests=5)
    app.add_middleware(RateLimitMiddleware, rate_limiter=test_limiter)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # Make request
    response = client.get("/test", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    
    # Check rate limit headers
    assert "X-RateLimit-Limit" in response.headers
    assert response.headers["X-RateLimit-Limit"] == "10"
    
    assert "X-RateLimit-Remaining" in response.headers
    remaining = int(response.headers["X-RateLimit-Remaining"])
    assert 0 <= remaining <= 10


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_handles_missing_client_info():
    """
    Test that rate limiter handles requests with missing client information.
    
    **Validates: Requirement 5.6**
    """
    limiter = RateLimiter(requests_per_minute=10, max_concurrent_requests=5)
    
    # Create mock request with no API key and no client
    request = Mock(spec=Request)
    request.headers.get.return_value = None
    request.client = None
    
    # Should still work (uses "unknown" as identifier)
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_slot_release_idempotent():
    """
    Test that releasing concurrent slots multiple times doesn't cause issues.
    
    **Validates: Requirement 7.3**
    """
    limiter = RateLimiter(requests_per_minute=10, max_concurrent_requests=5)
    
    # Create mock request
    request = Mock(spec=Request)
    request.headers.get.return_value = "test-api-key"
    request.client = Mock()
    request.client.host = "127.0.0.1"
    
    # Check rate limit (increments concurrent count)
    is_allowed, error_msg = await limiter.check_rate_limit(request)
    assert is_allowed is True
    
    # Release slot multiple times
    await limiter.release_concurrent_slot(request)
    await limiter.release_concurrent_slot(request)
    await limiter.release_concurrent_slot(request)
    
    # Should not cause negative count or errors
    client_id = limiter._get_client_identifier(request)
    assert limiter._concurrent_requests[client_id] == 0


def test_rate_limiter_initialization_from_settings():
    """
    Test that rate limiter can be initialized from settings.
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    limiter = RateLimiter(
        requests_per_minute=settings.rate_limit_per_minute,
        max_concurrent_requests=settings.max_concurrent_requests
    )
    
    assert limiter.requests_per_minute == settings.rate_limit_per_minute
    assert limiter.max_concurrent_requests == settings.max_concurrent_requests
