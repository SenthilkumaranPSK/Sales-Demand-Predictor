"""
Rate limiting middleware for the Demand Forecasting System API.

This module provides rate limiting functionality to protect the API from
excessive requests and ensure fair resource allocation.

**Validates: Requirements 5.6, 7.2, 7.3**
"""

import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import settings
from src.utils.logging_config import logger
from src.api.exceptions import RateLimitError


class RateLimiter:
    """
    In-memory rate limiter with per-minute request tracking and concurrent request limiting.
    
    Tracks:
    - Requests per minute per API key/IP address
    - Current concurrent requests per API key/IP address
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    
    def __init__(
        self,
        requests_per_minute: int = 1000,
        max_concurrent_requests: int = 100
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute per key/IP
            max_concurrent_requests: Maximum concurrent requests per key/IP
        """
        self.requests_per_minute = requests_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        
        # Track request timestamps per key (sliding window)
        # Key: (api_key or ip_address) -> deque of timestamps
        self._request_timestamps: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Track concurrent requests per key
        # Key: (api_key or ip_address) -> count
        self._concurrent_requests: Dict[str, int] = defaultdict(int)
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        logger.info(
            f"Rate limiter initialized: {requests_per_minute} req/min, "
            f"{max_concurrent_requests} concurrent"
        )
    
    def _get_client_identifier(self, request: Request) -> str:
        """
        Get unique identifier for the client (API key or IP address).
        
        Prefers API key if available, falls back to IP address.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get API key from header
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to IP address
        # Check for X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Use direct client IP
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    def _clean_old_timestamps(self, timestamps: deque, current_time: float) -> None:
        """
        Remove timestamps older than 1 minute from the deque.
        
        Args:
            timestamps: Deque of request timestamps
            current_time: Current time in seconds since epoch
        """
        cutoff_time = current_time - 60.0  # 60 seconds = 1 minute
        
        # Remove old timestamps from the left (oldest)
        while timestamps and timestamps[0] < cutoff_time:
            timestamps.popleft()
    
    async def check_rate_limit(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (is_allowed, error_message)
            - is_allowed: True if request is allowed, False if rate limited
            - error_message: Error message if rate limited, None otherwise
        """
        async with self._lock:
            client_id = self._get_client_identifier(request)
            current_time = time.time()
            
            # Check concurrent request limit
            concurrent_count = self._concurrent_requests[client_id]
            if concurrent_count >= self.max_concurrent_requests:
                error_msg = (
                    f"Concurrent request limit exceeded. "
                    f"Maximum {self.max_concurrent_requests} concurrent requests allowed."
                )
                logger.warning(
                    f"Rate limit exceeded (concurrent): client={client_id}, "
                    f"concurrent={concurrent_count}"
                )
                return False, error_msg
            
            # Get request timestamps for this client
            timestamps = self._request_timestamps[client_id]
            
            # Clean old timestamps (older than 1 minute)
            self._clean_old_timestamps(timestamps, current_time)
            
            # Check requests per minute limit
            request_count = len(timestamps)
            if request_count >= self.requests_per_minute:
                # Calculate time until oldest request expires
                oldest_timestamp = timestamps[0]
                time_until_reset = 60.0 - (current_time - oldest_timestamp)
                
                error_msg = (
                    f"Rate limit exceeded. "
                    f"Maximum {self.requests_per_minute} requests per minute allowed. "
                    f"Try again in {int(time_until_reset) + 1} seconds."
                )
                logger.warning(
                    f"Rate limit exceeded (per-minute): client={client_id}, "
                    f"requests={request_count}, reset_in={time_until_reset:.1f}s"
                )
                return False, error_msg
            
            # Request is allowed - record it
            timestamps.append(current_time)
            self._concurrent_requests[client_id] += 1
            
            logger.debug(
                f"Rate limit check passed: client={client_id}, "
                f"requests={request_count + 1}/{self.requests_per_minute}, "
                f"concurrent={concurrent_count + 1}/{self.max_concurrent_requests}"
            )
            
            return True, None
    
    async def release_concurrent_slot(self, request: Request) -> None:
        """
        Release a concurrent request slot for the client.
        
        Should be called when request processing completes.
        
        Args:
            request: FastAPI request object
        """
        async with self._lock:
            client_id = self._get_client_identifier(request)
            
            # Decrement concurrent request count
            if self._concurrent_requests[client_id] > 0:
                self._concurrent_requests[client_id] -= 1
            
            logger.debug(
                f"Released concurrent slot: client={client_id}, "
                f"concurrent={self._concurrent_requests[client_id]}"
            )
    
    def get_stats(self, client_id: Optional[str] = None) -> Dict:
        """
        Get rate limiter statistics.
        
        Args:
            client_id: Optional client identifier to get stats for specific client
            
        Returns:
            Dictionary with rate limiter statistics
        """
        if client_id:
            current_time = time.time()
            timestamps = self._request_timestamps.get(client_id, deque())
            
            # Clean old timestamps
            self._clean_old_timestamps(timestamps, current_time)
            
            return {
                "client_id": client_id,
                "requests_last_minute": len(timestamps),
                "requests_per_minute_limit": self.requests_per_minute,
                "concurrent_requests": self._concurrent_requests.get(client_id, 0),
                "max_concurrent_requests": self.max_concurrent_requests
            }
        else:
            # Return global stats
            return {
                "total_clients": len(self._request_timestamps),
                "requests_per_minute_limit": self.requests_per_minute,
                "max_concurrent_requests": self.max_concurrent_requests,
                "active_clients": sum(
                    1 for count in self._concurrent_requests.values() if count > 0
                )
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies rate limiting to all requests and returns HTTP 429 when limits exceeded.
    
    **Validates: Requirements 5.6, 7.2, 7.3**
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: ASGI application
            rate_limiter: RateLimiter instance (creates default if None)
        """
        super().__init__(app)
        
        if rate_limiter is None:
            # Create default rate limiter from settings
            rate_limiter = RateLimiter(
                requests_per_minute=settings.rate_limit_per_minute,
                max_concurrent_requests=settings.max_concurrent_requests
            )
        
        self.rate_limiter = rate_limiter
        logger.info("Rate limit middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response (either from handler or rate limit error)
        """
        # Skip rate limiting for health check endpoint
        if request.url.path == "/api/v1/health":
            return await call_next(request)
        
        # Check rate limit
        is_allowed, error_message = await self.rate_limiter.check_rate_limit(request)
        
        if not is_allowed:
            # Rate limit exceeded - return 429
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": error_message,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                },
                headers={
                    "Retry-After": "60",  # Suggest retry after 60 seconds
                    "X-RateLimit-Limit": str(self.rate_limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            client_id = self.rate_limiter._get_client_identifier(request)
            stats = self.rate_limiter.get_stats(client_id)
            
            response.headers["X-RateLimit-Limit"] = str(
                self.rate_limiter.requests_per_minute
            )
            response.headers["X-RateLimit-Remaining"] = str(
                self.rate_limiter.requests_per_minute - stats["requests_last_minute"]
            )
            
            return response
            
        finally:
            # Always release concurrent slot when request completes
            await self.rate_limiter.release_concurrent_slot(request)


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=settings.rate_limit_per_minute,
    max_concurrent_requests=settings.max_concurrent_requests
)
