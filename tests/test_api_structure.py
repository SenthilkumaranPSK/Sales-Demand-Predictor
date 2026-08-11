"""
Tests for FastAPI application structure.

This module tests the FastAPI application structure including:
- Pydantic request/response models
- Dependency injection
- OpenAPI documentation configuration

Validates: Requirements 5.1, 5.2, 5.3
"""
import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from src.api.main import (
    app,
    ForecastRequest,
    ForecastResponse,
    ConfidenceIntervalModel,
    get_forecasting_engine,
    get_model_registry
)
from src.inference.forecasting_engine import ForecastingEngine
from src.registry.model_registry import ModelRegistry


class TestPydanticModels:
    """Test Pydantic request/response models."""
    
    def test_confidence_interval_model_creation(self):
        """Test ConfidenceIntervalModel can be created with valid data."""
        ci = ConfidenceIntervalModel(
            level="80%",
            lower=[95.2, 98.1, 102.3],
            upper=[105.8, 108.9, 112.7]
        )
        
        assert ci.level == "80%"
        assert len(ci.lower) == 3
        assert len(ci.upper) == 3
        assert ci.lower[0] == 95.2
        assert ci.upper[0] == 105.8
    
    def test_forecast_request_validation(self):
        """Test ForecastRequest validates required fields."""
        # Valid request
        request = ForecastRequest(
            product_id="PROD-12345",
            forecast_horizon=30
        )
        
        assert request.product_id == "PROD-12345"
        assert request.forecast_horizon == 30
        assert request.model_id is None
        assert request.include_benchmark is False
        assert request.future_features is None
    
    def test_forecast_request_horizon_validation(self):
        """Test ForecastRequest validates forecast_horizon range (1-90)."""
        # Valid horizons
        request1 = ForecastRequest(product_id="PROD-123", forecast_horizon=1)
        assert request1.forecast_horizon == 1
        
        request2 = ForecastRequest(product_id="PROD-123", forecast_horizon=90)
        assert request2.forecast_horizon == 90
        
        # Invalid horizons
        with pytest.raises(ValueError):
            ForecastRequest(product_id="PROD-123", forecast_horizon=0)
        
        with pytest.raises(ValueError):
            ForecastRequest(product_id="PROD-123", forecast_horizon=91)
        
        with pytest.raises(ValueError):
            ForecastRequest(product_id="PROD-123", forecast_horizon=-5)
    
    def test_forecast_request_product_id_validation(self):
        """Test ForecastRequest validates product_id is non-empty."""
        # Valid product_id
        request = ForecastRequest(product_id="PROD-123", forecast_horizon=30)
        assert request.product_id == "PROD-123"
        
        # Empty product_id should fail
        with pytest.raises(ValueError):
            ForecastRequest(product_id="", forecast_horizon=30)
    
    def test_forecast_request_with_optional_fields(self):
        """Test ForecastRequest with all optional fields."""
        request = ForecastRequest(
            product_id="PROD-12345",
            forecast_horizon=30,
            model_id="model_123",
            include_benchmark=True,
            future_features={
                "holidays": [False] * 30,
                "prices": [19.99] * 30
            }
        )
        
        assert request.model_id == "model_123"
        assert request.include_benchmark is True
        assert request.future_features is not None
        assert len(request.future_features["holidays"]) == 30
    
    def test_forecast_response_structure(self):
        """Test ForecastResponse has all required fields."""
        response = ForecastResponse(
            forecast_id="forecast_123",
            product_id="PROD-12345",
            model_id="model_123",
            timestamps=[datetime(2025, 1, 16), datetime(2025, 1, 17)],
            predictions=[100.5, 105.2],
            confidence_intervals={
                "50%": ConfidenceIntervalModel(
                    level="50%",
                    lower=[95.2, 99.8],
                    upper=[105.8, 110.6]
                ),
                "80%": ConfidenceIntervalModel(
                    level="80%",
                    lower=[90.3, 94.9],
                    upper=[110.7, 115.5]
                ),
                "90%": ConfidenceIntervalModel(
                    level="90%",
                    lower=[85.4, 90.0],
                    upper=[115.6, 120.4]
                )
            },
            metadata={
                "algorithm": "random_forest",
                "model_version": 1
            }
        )
        
        assert response.forecast_id == "forecast_123"
        assert response.product_id == "PROD-12345"
        assert response.model_id == "model_123"
        assert len(response.timestamps) == 2
        assert len(response.predictions) == 2
        assert len(response.confidence_intervals) == 3
        assert "50%" in response.confidence_intervals
        assert "80%" in response.confidence_intervals
        assert "90%" in response.confidence_intervals
        assert response.benchmark is None
        assert response.metadata["algorithm"] == "random_forest"


class TestDependencyInjection:
    """Test dependency injection functions."""
    
    def test_get_forecasting_engine_returns_instance(self):
        """Test get_forecasting_engine returns ForecastingEngine instance."""
        engine = get_forecasting_engine()
        
        assert engine is not None
        assert isinstance(engine, ForecastingEngine)
    
    def test_get_model_registry_returns_instance(self):
        """Test get_model_registry returns ModelRegistry instance."""
        registry = get_model_registry()
        
        assert registry is not None
        assert isinstance(registry, ModelRegistry)
    
    def test_dependency_injection_returns_same_instance(self):
        """Test dependency injection returns the same global instance."""
        engine1 = get_forecasting_engine()
        engine2 = get_forecasting_engine()
        
        # Should return the same instance (singleton pattern)
        assert engine1 is engine2
        
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        
        assert registry1 is registry2


class TestOpenAPIConfiguration:
    """Test OpenAPI documentation configuration."""
    
    def test_openapi_schema_generated(self):
        """Test OpenAPI schema is generated correctly."""
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Check basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check API metadata
        assert schema["info"]["title"] == "Demand Forecasting System API"
        assert schema["info"]["version"] == "1.0.0"
        assert "description" in schema["info"]
    
    def test_openapi_tags_configured(self):
        """Test OpenAPI tags are configured."""
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check tags are defined
        assert "tags" in schema
        tag_names = [tag["name"] for tag in schema["tags"]]
        
        assert "Forecasting" in tag_names
        assert "Models" in tag_names
        assert "Health" in tag_names
        assert "Root" in tag_names
    
    def test_openapi_components_include_models(self):
        """Test OpenAPI schema includes Pydantic models."""
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check components/schemas include our models
        assert "components" in schema
        assert "schemas" in schema["components"]
        
        schemas = schema["components"]["schemas"]
        
        # Check health check model is in the schema (it's used in an endpoint)
        assert "HealthCheckResponse" in schemas
        
        # Note: ForecastRequest and ForecastResponse will be added when
        # the forecast endpoint is implemented in Task 12.2
    
    def test_forecast_request_schema_validation(self):
        """Test ForecastRequest model has correct validation rules."""
        # Test the model directly since it's not yet used in an endpoint
        
        # Valid request
        request = ForecastRequest(
            product_id="PROD-123",
            forecast_horizon=30
        )
        assert request.product_id == "PROD-123"
        assert request.forecast_horizon == 30
        
        # Test validation rules
        with pytest.raises(ValueError):
            # forecast_horizon must be >= 1
            ForecastRequest(product_id="PROD-123", forecast_horizon=0)
        
        with pytest.raises(ValueError):
            # forecast_horizon must be <= 90
            ForecastRequest(product_id="PROD-123", forecast_horizon=91)
        
        with pytest.raises(ValueError):
            # product_id must not be empty
            ForecastRequest(product_id="", forecast_horizon=30)


class TestCORSConfiguration:
    """Test CORS middleware configuration."""
    
    def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        client = TestClient(app)
        
        # Make a request with Origin header
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Check CORS headers are present
        # CORS middleware reflects the origin when allow_origins=["*"]
        assert "access-control-allow-origin" in response.headers
        # The origin should be reflected back
        assert response.headers["access-control-allow-origin"] in ["*", "http://localhost:3000"]


class TestApplicationMetadata:
    """Test FastAPI application metadata."""
    
    def test_app_title(self):
        """Test application title is set correctly."""
        assert app.title == "Demand Forecasting System API"
    
    def test_app_version(self):
        """Test application version is set correctly."""
        assert app.version == "1.0.0"
    
    def test_app_docs_url(self):
        """Test docs URL is configured."""
        assert app.docs_url == "/api/docs"
    
    def test_app_redoc_url(self):
        """Test ReDoc URL is configured."""
        assert app.redoc_url == "/api/redoc"
    
    def test_docs_endpoint_accessible(self):
        """Test Swagger UI docs endpoint is accessible."""
        client = TestClient(app)
        
        response = client.get("/api/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint_accessible(self):
        """Test ReDoc endpoint is accessible."""
        client = TestClient(app)
        
        response = client.get("/api/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
