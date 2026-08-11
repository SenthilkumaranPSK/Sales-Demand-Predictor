"""API authentication middleware and dependencies."""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from config.settings import settings
from src.utils.logging_config import logger


# API Key header security scheme
api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False  # We'll handle errors manually for better control
)


def get_valid_api_keys() -> set:
    """
    Get valid API keys from configuration.
    
    In production, this would integrate with AWS API Gateway or
    a database of valid keys. For now, we support environment-based
    configuration.
    
    Returns:
        Set of valid API key strings
    """
    # Check for API keys in settings
    # Support both single key and comma-separated list
    api_keys_str = getattr(settings, 'api_keys', None)
    
    if not api_keys_str:
        # Default development key if none configured
        logger.warning(
            "No API keys configured. Using default development key. "
            "Set API_KEYS environment variable for production."
        )
        return {"dev-api-key-12345"}
    
    # Parse comma-separated keys
    keys = {key.strip() for key in api_keys_str.split(',') if key.strip()}
    
    if not keys:
        logger.warning("API_KEYS configured but empty. Using default development key.")
        return {"dev-api-key-12345"}
    
    return keys


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.
    
    This dependency can be used to protect endpoints that require authentication.
    
    **Validates: Requirement 5.5**
    
    Args:
        api_key: API key from X-API-Key header (injected by FastAPI)
        
    Returns:
        The validated API key string
        
    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    # Check if API key is provided
    if not api_key:
        logger.warning("API request rejected: Missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "MISSING_API_KEY",
                    "message": f"API key required. Provide valid API key in {settings.api_key_header} header.",
                    "timestamp": None  # Will be set by error handler
                }
            },
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Validate API key
    valid_keys = get_valid_api_keys()
    
    if api_key not in valid_keys:
        logger.warning(f"API request rejected: Invalid API key (key={api_key[:8]}...)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "INVALID_API_KEY",
                    "message": "Invalid API key. Please check your credentials.",
                    "timestamp": None  # Will be set by error handler
                }
            },
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    logger.debug(f"API key validated successfully (key={api_key[:8]}...)")
    return api_key
