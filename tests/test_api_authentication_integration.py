"""
Integration tests for API authentication on endpoints.

**Validates: Requirement 5.5**
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.api.main import app
from config.settings import settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_forecasting_engine():
    """Mock forecasting engine for testing."""
    with patch('src.api.main.forecasting_engine') as mock_engine:
        # Create mock forecast result
        mock_result = MagicMock()
        mock_result.product_id = "PROD-123"
        mock_result.model_id = "model_123"
        mock_result.timestamps = [datetime(2025, 1, 16), datetime(2025, 1, 17)]
        mock_result.predictions = [100.0, 105.0]
        mock_result.confidence_intervals = {
            "50%": MagicMock(level="50%", lower=[95.0, 100.0], upper=[105.0, 110.0]),
            "80%": MagicMock(level="80%", lower=[90.0, 95.0], upper=[110.0, 115.0]),
            "90%": MagicMock(level="90%", lower=[85.0, 90.0], upper=[115.0, 120.0])
        }
        mock_result.metadata = {"algorithm": "test"}
        
        mock_engine.generate_forecast.return_value = mock_result
        yield mock_engine


@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    with patch('src.api.main.model_registry') as mock_registry:
        mock_registry.get_latest_model.return_value = ("model_123", MagicMock())
        yield mock_registry


class TestHealthCheckEndpoint:
    """Test health check endpoint does not require authentication."""
    
    def test_health_check_without_api_key(self, client):
        """Health check should work without API key."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_health_check_with_invalid_api_key(self, client):
        """Health check should work even with invalid API key."""
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 200
    
    def test_health_check_with_valid_api_key(self, client):
        """Health check should work with valid API key."""
        with patch.object(settings, 'api_keys', 'test-key'):
            response = client.get(
                "/api/v1/health",
                headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200


class TestRootEndpoint:
    """Test root endpoint does not require authentication."""
    
    def test_root_without_api_key(self, client):
        """Root endpoint should work without API key."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_root_with_invalid_api_key(self, client):
        """Root endpoint should work even with invalid API key."""
        response = client.get(
            "/",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 200
    
    def test_root_with_valid_api_key(self, client):
        """Root endpoint should work with valid API key."""
        with patch.object(settings, 'api_keys', 'test-key'):
            response = client.get(
                "/",
                headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200


class TestForecastEndpointAuthentication:
    """Test forecast endpoint requires authentication."""
    
    def test_forecast_without_api_key_returns_401(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should return 401 without API key."""
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30
            }
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "MISSING_API_KEY"
        assert settings.api_key_header in data["detail"]["error"]["message"]
    
    def test_forecast_with_empty_api_key_returns_401(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should return 401 with empty API key."""
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30
            },
            headers={"X-API-Key": ""}
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "MISSING_API_KEY"
    
    def test_forecast_with_invalid_api_key_returns_401(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should return 401 with invalid API key."""
        with patch.object(settings, 'api_keys', 'valid-key-123'):
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "invalid-key"}
            )
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "error" in data["detail"]
            assert data["detail"]["error"]["code"] == "INVALID_API_KEY"
    
    def test_forecast_with_valid_api_key_succeeds(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should succeed with valid API key."""
        with patch.object(settings, 'api_keys', 'valid-key-123'):
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "valid-key-123"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "forecast_id" in data
            assert "product_id" in data
            assert data["product_id"] == "PROD-123"
    
    def test_forecast_with_default_development_key_succeeds(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should succeed with default development key when no keys configured."""
        with patch.object(settings, 'api_keys', None):
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "dev-api-key-12345"}
            )
            assert response.status_code == 200
    
    def test_forecast_with_one_of_multiple_valid_keys_succeeds(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Forecast endpoint should succeed with any of multiple valid keys."""
        with patch.object(settings, 'api_keys', 'key1,key2,key3'):
            # Test with key1
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "key1"}
            )
            assert response.status_code == 200
            
            # Test with key2
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "key2"}
            )
            assert response.status_code == 200
            
            # Test with key3
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "key3"}
            )
            assert response.status_code == 200
    
    def test_401_response_includes_www_authenticate_header(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """401 response should include WWW-Authenticate header."""
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30
            }
        )
        assert response.status_code == 401
        assert "www-authenticate" in response.headers
        assert response.headers["www-authenticate"] == "ApiKey"


class TestApiKeyHeaderConfiguration:
    """Test that API key header name is configurable."""
    
    def test_uses_configured_header_name(
        self, client, mock_forecasting_engine, mock_model_registry
    ):
        """Should use the configured API key header name."""
        # Verify default header name is X-API-Key
        assert settings.api_key_header == "X-API-Key"
        
        with patch.object(settings, 'api_keys', 'test-key'):
            # Should work with X-API-Key header
            response = client.post(
                "/api/v1/forecast",
                json={
                    "product_id": "PROD-123",
                    "forecast_horizon": 30
                },
                headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200
