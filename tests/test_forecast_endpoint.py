"""
Tests for the forecast endpoint.

**Validates: Requirements 5.1, 5.2, 5.3, 5.5, 7.2**
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from src.api.main import app
from src.inference.forecasting_engine import ForecastResult, ConfidenceInterval
from src.registry.model_registry import ModelMetadata
from config.settings import settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def api_key_header():
    """Provide valid API key header for authenticated requests."""
    # Use default development key for tests
    with patch.object(settings, 'api_keys', None):
        yield {"X-API-Key": "dev-api-key-12345"}


@pytest.fixture
def mock_forecast_result():
    """Create mock forecast result."""
    return ForecastResult(
        model_id="model_test_v1",
        product_id="PROD-123",
        timestamps=[
            datetime(2025, 1, 16),
            datetime(2025, 1, 17),
            datetime(2025, 1, 18)
        ],
        predictions=[100.0, 105.0, 110.0],
        confidence_intervals={
            "50%": ConfidenceInterval(
                level="50%",
                lower=[95.0, 100.0, 105.0],
                upper=[105.0, 110.0, 115.0]
            ),
            "80%": ConfidenceInterval(
                level="80%",
                lower=[90.0, 95.0, 100.0],
                upper=[110.0, 115.0, 120.0]
            ),
            "90%": ConfidenceInterval(
                level="90%",
                lower=[85.0, 90.0, 95.0],
                upper=[115.0, 120.0, 125.0]
            )
        },
        metadata={
            "algorithm": "random_forest",
            "model_version": 1,
            "training_mae": 5.2,
            "training_rmse": 7.8,
            "training_mape": 4.5
        }
    )


@pytest.fixture
def mock_model_metadata():
    """Create mock model metadata."""
    return ModelMetadata(
        model_id="model_test_v1",
        product_id="PROD-123",
        model_type="custom",
        version=1,
        artifact_path="s3://bucket/model",
        training_dataset_id="dataset_123",
        mae=5.2,
        rmse=7.8,
        mape=4.5,
        hyperparameters={"n_estimators": 100},
        created_at=datetime(2025, 1, 15),
        forecast_horizon=30
    )


def test_forecast_endpoint_with_model_id(client, api_key_header, mock_forecast_result):
    """
    Test forecast endpoint with explicit model_id.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.5**
    """
    with patch("src.api.main.forecasting_engine") as mock_engine:
        mock_engine.generate_forecast.return_value = mock_forecast_result
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 3,
                "model_id": "model_test_v1",
                "include_benchmark": False
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "forecast_id" in data
        assert data["product_id"] == "PROD-123"
        assert data["model_id"] == "model_test_v1"
        assert len(data["predictions"]) == 3
        assert data["predictions"] == [100.0, 105.0, 110.0]
        
        # Verify confidence intervals
        assert "50%" in data["confidence_intervals"]
        assert "80%" in data["confidence_intervals"]
        assert "90%" in data["confidence_intervals"]
        
        # Verify metadata
        assert "algorithm" in data["metadata"]
        assert data["metadata"]["algorithm"] == "random_forest"
        
        # Verify engine was called correctly
        mock_engine.generate_forecast.assert_called_once()
        call_args = mock_engine.generate_forecast.call_args
        assert call_args[1]["model_id"] == "model_test_v1"
        assert call_args[1]["forecast_horizon"] == 3


def test_forecast_endpoint_without_model_id(client, api_key_header, mock_forecast_result, mock_model_metadata):
    """
    Test forecast endpoint without model_id (uses latest custom model).
    
    **Validates: Requirements 5.1, 5.2, 5.5**
    """
    with patch("src.api.main.forecasting_engine") as mock_engine, \
         patch("src.api.main.model_registry") as mock_registry:
        
        mock_engine.generate_forecast.return_value = mock_forecast_result
        mock_registry.get_latest_model.return_value = ("model_test_v1", mock_model_metadata)
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 3,
                "include_benchmark": False
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify latest model was retrieved
        mock_registry.get_latest_model.assert_called_once_with(
            product_id="PROD-123",
            model_type="custom"
        )
        
        # Verify forecast was generated
        assert data["model_id"] == "model_test_v1"


def test_forecast_endpoint_with_benchmark(client, api_key_header, mock_forecast_result, mock_model_metadata):
    """
    Test forecast endpoint with benchmark model included.
    
    **Validates: Requirements 5.2, 5.3, 5.5**
    """
    # Create benchmark forecast result
    benchmark_result = ForecastResult(
        model_id="model_benchmark_v1",
        product_id="PROD-123",
        timestamps=[
            datetime(2025, 1, 16),
            datetime(2025, 1, 17),
            datetime(2025, 1, 18)
        ],
        predictions=[98.0, 103.0, 108.0],
        confidence_intervals={
            "50%": ConfidenceInterval(
                level="50%",
                lower=[93.0, 98.0, 103.0],
                upper=[103.0, 108.0, 113.0]
            ),
            "80%": ConfidenceInterval(
                level="80%",
                lower=[88.0, 93.0, 98.0],
                upper=[108.0, 113.0, 118.0]
            ),
            "90%": ConfidenceInterval(
                level="90%",
                lower=[83.0, 88.0, 93.0],
                upper=[113.0, 118.0, 123.0]
            )
        },
        metadata={
            "algorithm": "amazon_forecast",
            "model_version": 1
        }
    )
    
    benchmark_metadata = ModelMetadata(
        model_id="model_benchmark_v1",
        product_id="PROD-123",
        model_type="forecast",
        version=1,
        artifact_path="s3://bucket/benchmark",
        training_dataset_id="dataset_123",
        mae=6.0,
        rmse=8.5,
        mape=5.0,
        hyperparameters={},
        created_at=datetime(2025, 1, 15),
        forecast_horizon=30
    )
    
    with patch("src.api.main.forecasting_engine") as mock_engine, \
         patch("src.api.main.model_registry") as mock_registry:
        
        # Mock custom model
        mock_registry.get_latest_model.side_effect = [
            ("model_test_v1", mock_model_metadata),  # First call for custom
            ("model_benchmark_v1", benchmark_metadata)  # Second call for benchmark
        ]
        
        # Mock forecast generation
        mock_engine.generate_forecast.side_effect = [
            mock_forecast_result,  # Custom forecast
            benchmark_result  # Benchmark forecast
        ]
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 3,
                "include_benchmark": True
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify benchmark is included
        assert data["benchmark"] is not None
        assert data["benchmark"]["model_id"] == "model_benchmark_v1"
        assert data["benchmark"]["predictions"] == [98.0, 103.0, 108.0]


def test_forecast_endpoint_invalid_horizon(client, api_key_header):
    """
    Test forecast endpoint with invalid forecast horizon.
    
    **Validates: Requirements 5.4, 5.5, 10.4**
    """
    response = client.post(
        "/api/v1/forecast",
        json={
            "product_id": "PROD-123",
            "forecast_horizon": 120,  # Invalid: > 90
            "model_id": "model_test_v1"
        },
        headers=api_key_header
    )
    
    # Should return 400 (validation error handled by error handler)
    assert response.status_code == 400


def test_forecast_endpoint_missing_product_id(client, api_key_header):
    """
    Test forecast endpoint with missing product_id.
    
    **Validates: Requirements 5.4, 5.5**
    """
    response = client.post(
        "/api/v1/forecast",
        json={
            "forecast_horizon": 30,
            "model_id": "model_test_v1"
        },
        headers=api_key_header
    )
    
    # Should return 400 (validation error handled by error handler)
    assert response.status_code == 400


def test_forecast_endpoint_model_not_found(client, api_key_header):
    """
    Test forecast endpoint when no model exists for product.
    
    **Validates: Requirements 4.5, 5.4, 5.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.get_latest_model.side_effect = ValueError("No model found")
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-999",
                "forecast_horizon": 30
            },
            headers=api_key_header
        )
        
        # ModelNotFoundError is caught and handled by error handler
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MODEL_NOT_FOUND"


def test_forecast_endpoint_forecast_generation_error(client, api_key_header, mock_model_metadata):
    """
    Test forecast endpoint when forecast generation fails.
    
    **Validates: Requirements 5.5, 10.2**
    """
    with patch("src.api.main.forecasting_engine") as mock_engine, \
         patch("src.api.main.model_registry") as mock_registry:
        
        mock_registry.get_latest_model.return_value = ("model_test_v1", mock_model_metadata)
        mock_engine.generate_forecast.side_effect = RuntimeError("Model loading failed")
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30
            },
            headers=api_key_header
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "FORECAST_ERROR"


def test_forecast_endpoint_with_future_features(client, api_key_header, mock_forecast_result):
    """
    Test forecast endpoint with future features.
    
    **Validates: Requirements 5.1, 5.5, 9.4**
    """
    with patch("src.api.main.forecasting_engine") as mock_engine:
        mock_engine.generate_forecast.return_value = mock_forecast_result
        
        future_features = {
            "holidays": [False, False, True],
            "prices": [19.99, 19.99, 17.99]
        }
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 3,
                "model_id": "model_test_v1",
                "future_features": future_features
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        
        # Verify future features were passed to engine
        call_args = mock_engine.generate_forecast.call_args
        assert call_args[1]["future_features"] == future_features


def test_forecast_endpoint_response_time(client, api_key_header, mock_forecast_result):
    """
    Test forecast endpoint response time target.
    
    **Validates: Requirements 5.5, 7.2**
    
    Note: This test verifies the endpoint completes quickly with mocked engine.
    Real performance testing should be done with actual models and load testing.
    """
    with patch("src.api.main.forecasting_engine") as mock_engine:
        mock_engine.generate_forecast.return_value = mock_forecast_result
        
        import time
        start = time.time()
        
        response = client.post(
            "/api/v1/forecast",
            json={
                "product_id": "PROD-123",
                "forecast_horizon": 30,
                "model_id": "model_test_v1"
            },
            headers=api_key_header
        )
        
        elapsed = time.time() - start
        
        assert response.status_code == 200
        # With mocked engine, should be very fast (< 1 second)
        assert elapsed < 1.0
