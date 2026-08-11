"""
Unit tests for API authentication.

**Validates: Requirement 5.5**
"""
import pytest
from fastapi import HTTPException, status
from unittest.mock import patch, MagicMock

from src.api.auth import verify_api_key, get_valid_api_keys
from config.settings import settings


class TestGetValidApiKeys:
    """Test suite for get_valid_api_keys function."""
    
    def test_returns_default_key_when_no_keys_configured(self):
        """Should return default development key when no API keys configured."""
        with patch.object(settings, 'api_keys', None):
            keys = get_valid_api_keys()
            assert "dev-api-key-12345" in keys
            assert len(keys) == 1
    
    def test_returns_default_key_when_empty_string(self):
        """Should return default development key when API_KEYS is empty string."""
        with patch.object(settings, 'api_keys', ''):
            keys = get_valid_api_keys()
            assert "dev-api-key-12345" in keys
            assert len(keys) == 1
    
    def test_returns_default_key_when_only_whitespace(self):
        """Should return default development key when API_KEYS contains only whitespace."""
        with patch.object(settings, 'api_keys', '   ,  , '):
            keys = get_valid_api_keys()
            assert "dev-api-key-12345" in keys
            assert len(keys) == 1
    
    def test_parses_single_api_key(self):
        """Should parse single API key correctly."""
        with patch.object(settings, 'api_keys', 'test-key-123'):
            keys = get_valid_api_keys()
            assert "test-key-123" in keys
            assert len(keys) == 1
    
    def test_parses_multiple_comma_separated_keys(self):
        """Should parse multiple comma-separated API keys."""
        with patch.object(settings, 'api_keys', 'key1,key2,key3'):
            keys = get_valid_api_keys()
            assert "key1" in keys
            assert "key2" in keys
            assert "key3" in keys
            assert len(keys) == 3
    
    def test_strips_whitespace_from_keys(self):
        """Should strip whitespace from API keys."""
        with patch.object(settings, 'api_keys', ' key1 , key2  ,  key3'):
            keys = get_valid_api_keys()
            assert "key1" in keys
            assert "key2" in keys
            assert "key3" in keys
            assert " key1 " not in keys
            assert len(keys) == 3
    
    def test_handles_duplicate_keys(self):
        """Should deduplicate API keys (set behavior)."""
        with patch.object(settings, 'api_keys', 'key1,key2,key1,key3,key2'):
            keys = get_valid_api_keys()
            assert "key1" in keys
            assert "key2" in keys
            assert "key3" in keys
            assert len(keys) == 3


class TestVerifyApiKey:
    """Test suite for verify_api_key dependency."""
    
    @pytest.mark.asyncio
    async def test_raises_401_when_api_key_is_none(self):
        """Should raise 401 HTTPException when API key is None."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key=None)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "MISSING_API_KEY" in str(exc_info.value.detail)
        assert settings.api_key_header in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_raises_401_when_api_key_is_empty_string(self):
        """Should raise 401 HTTPException when API key is empty string."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key='')
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "MISSING_API_KEY" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_raises_401_when_api_key_is_invalid(self):
        """Should raise 401 HTTPException when API key is not in valid keys."""
        with patch('src.api.auth.get_valid_api_keys', return_value={'valid-key-1', 'valid-key-2'}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key='invalid-key')
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "INVALID_API_KEY" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_returns_api_key_when_valid(self):
        """Should return API key when it is valid."""
        valid_key = 'valid-key-123'
        with patch('src.api.auth.get_valid_api_keys', return_value={valid_key}):
            result = await verify_api_key(api_key=valid_key)
            assert result == valid_key
    
    @pytest.mark.asyncio
    async def test_validates_against_multiple_keys(self):
        """Should validate API key against multiple valid keys."""
        valid_keys = {'key1', 'key2', 'key3'}
        with patch('src.api.auth.get_valid_api_keys', return_value=valid_keys):
            # Test each valid key
            for key in valid_keys:
                result = await verify_api_key(api_key=key)
                assert result == key
    
    @pytest.mark.asyncio
    async def test_includes_www_authenticate_header_on_401(self):
        """Should include WWW-Authenticate header in 401 responses."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key=None)
        
        assert exc_info.value.headers is not None
        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "ApiKey"
    
    @pytest.mark.asyncio
    async def test_error_detail_has_correct_structure(self):
        """Should return error detail with correct structure."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key='invalid')
        
        detail = exc_info.value.detail
        assert "error" in detail
        assert "code" in detail["error"]
        assert "message" in detail["error"]
        assert "timestamp" in detail["error"]


class TestApiKeyIntegration:
    """Integration tests for API key authentication."""
    
    @pytest.mark.asyncio
    async def test_default_development_key_works(self):
        """Should accept default development key when no keys configured."""
        with patch.object(settings, 'api_keys', None):
            result = await verify_api_key(api_key='dev-api-key-12345')
            assert result == 'dev-api-key-12345'
    
    @pytest.mark.asyncio
    async def test_configured_keys_override_default(self):
        """Should not accept default key when custom keys are configured."""
        with patch.object(settings, 'api_keys', 'custom-key-1,custom-key-2'):
            # Default key should not work
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key='dev-api-key-12345')
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            
            # Custom keys should work
            result = await verify_api_key(api_key='custom-key-1')
            assert result == 'custom-key-1'
