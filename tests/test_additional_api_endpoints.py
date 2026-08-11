"""
Tests for additional API endpoints: models listing, model metadata, and data ingestion.

**Validates: Requirement 10.5**
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

from src.api.main import app
from src.registry.model_registry import ModelMetadata
from src.data.ingestion import IngestionResult
from src.data.validation import ValidationResult
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
def mock_model_metadata_list():
    """Create list of mock model metadata."""
    return [
        ModelMetadata(
            model_id="model_prod123_v1",
            product_id="PROD-123",
            model_type="custom",
            version=1,
            artifact_path="s3://bucket/model1",
            training_dataset_id="dataset_123",
            mae=5.2,
            rmse=7.8,
            mape=4.5,
            hyperparameters={"n_estimators": 100},
            created_at=datetime(2025, 1, 15, 10, 30, 0),
            forecast_horizon=30
        ),
        ModelMetadata(
            model_id="model_prod123_v2",
            product_id="PROD-123",
            model_type="custom",
            version=2,
            artifact_path="s3://bucket/model2",
            training_dataset_id="dataset_124",
            mae=4.8,
            rmse=7.2,
            mape=4.1,
            hyperparameters={"n_estimators": 150},
            created_at=datetime(2025, 1, 16, 10, 30, 0),
            forecast_horizon=30
        ),
        ModelMetadata(
            model_id="model_prod456_v1",
            product_id="PROD-456",
            model_type="forecast",
            version=1,
            artifact_path="s3://bucket/model3",
            training_dataset_id="dataset_125",
            mae=6.0,
            rmse=8.5,
            mape=5.0,
            hyperparameters={},
            created_at=datetime(2025, 1, 15, 11, 30, 0),
            forecast_horizon=30
        )
    ]


@pytest.fixture
def mock_ingestion_result_success():
    """Create successful ingestion result."""
    return IngestionResult(
        success=True,
        record_count=1000,
        s3_path="s3://bucket/historical_data/dataset_20250115_123456",
        validation_result=ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            record_count=1000
        ),
        errors=[],
        ingestion_time_seconds=2.5
    )


@pytest.fixture
def mock_ingestion_result_failure():
    """Create failed ingestion result."""
    return IngestionResult(
        success=False,
        record_count=100,
        s3_path=None,
        validation_result=None,
        errors=["missing_column in timestamp: Required column 'timestamp' is missing"],
        ingestion_time_seconds=0.5
    )


# ============================================================================
# Tests for GET /api/v1/models
# ============================================================================


def test_list_models_all(client, api_key_header, mock_model_metadata_list):
    """
    Test listing all models without filters.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.return_value = mock_model_metadata_list
        
        response = client.get(
            "/api/v1/models",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "models" in data
        assert "total_count" in data
        assert data["total_count"] == 3
        assert len(data["models"]) == 3
        
        # Verify first model
        model1 = data["models"][0]
        assert model1["model_id"] == "model_prod123_v1"
        assert model1["product_id"] == "PROD-123"
        assert model1["model_type"] == "custom"
        assert model1["version"] == 1
        assert model1["mae"] == 5.2
        assert model1["rmse"] == 7.8
        assert model1["mape"] == 4.5
        assert model1["forecast_horizon"] == 30
        
        # Verify registry was called correctly
        mock_registry.list_models.assert_called_once_with(
            product_id=None,
            model_type=None
        )


def test_list_models_filter_by_product_id(client, api_key_header, mock_model_metadata_list):
    """
    Test listing models filtered by product_id.
    
    **Validates: Requirement 10.5**
    """
    # Filter to only PROD-123 models
    filtered_models = [m for m in mock_model_metadata_list if m.product_id == "PROD-123"]
    
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.return_value = filtered_models
        
        response = client.get(
            "/api/v1/models?product_id=PROD-123",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_count"] == 2
        assert len(data["models"]) == 2
        assert all(m["product_id"] == "PROD-123" for m in data["models"])
        
        # Verify registry was called with filter
        mock_registry.list_models.assert_called_once_with(
            product_id="PROD-123",
            model_type=None
        )


def test_list_models_filter_by_model_type(client, api_key_header, mock_model_metadata_list):
    """
    Test listing models filtered by model_type.
    
    **Validates: Requirement 10.5**
    """
    # Filter to only custom models
    filtered_models = [m for m in mock_model_metadata_list if m.model_type == "custom"]
    
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.return_value = filtered_models
        
        response = client.get(
            "/api/v1/models?model_type=custom",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_count"] == 2
        assert len(data["models"]) == 2
        assert all(m["model_type"] == "custom" for m in data["models"])
        
        # Verify registry was called with filter
        mock_registry.list_models.assert_called_once_with(
            product_id=None,
            model_type="custom"
        )


def test_list_models_filter_by_both(client, api_key_header, mock_model_metadata_list):
    """
    Test listing models filtered by both product_id and model_type.
    
    **Validates: Requirement 10.5**
    """
    # Filter to PROD-123 custom models
    filtered_models = [
        m for m in mock_model_metadata_list 
        if m.product_id == "PROD-123" and m.model_type == "custom"
    ]
    
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.return_value = filtered_models
        
        response = client.get(
            "/api/v1/models?product_id=PROD-123&model_type=custom",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_count"] == 2
        assert all(
            m["product_id"] == "PROD-123" and m["model_type"] == "custom" 
            for m in data["models"]
        )


def test_list_models_invalid_model_type(client, api_key_header):
    """
    Test listing models with invalid model_type.
    
    **Validates: Requirement 10.5**
    """
    response = client.get(
        "/api/v1/models?model_type=invalid",
        headers=api_key_header
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]
    assert data["detail"]["error"]["code"] == "VALIDATION_ERROR"


def test_list_models_empty_result(client, api_key_header):
    """
    Test listing models when no models exist.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.return_value = []
        
        response = client.get(
            "/api/v1/models",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_count"] == 0
        assert len(data["models"]) == 0


def test_list_models_no_auth(client):
    """
    Test listing models without authentication.
    
    **Validates: Requirement 10.5**
    """
    response = client.get("/api/v1/models")
    
    assert response.status_code == 401


def test_list_models_registry_error(client, api_key_header):
    """
    Test listing models when registry fails.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.list_models.side_effect = RuntimeError("Database error")
        
        response = client.get(
            "/api/v1/models",
            headers=api_key_header
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]


# ============================================================================
# Tests for GET /api/v1/models/{model_id}
# ============================================================================


def test_get_model_metadata_success(client, api_key_header, mock_model_metadata_list):
    """
    Test getting model metadata by ID.
    
    **Validates: Requirement 10.5**
    """
    model_metadata = mock_model_metadata_list[0]
    
    with patch("src.api.main.model_registry") as mock_registry:
        # get_model returns (artifact, metadata)
        mock_registry.get_model.return_value = (b"model_artifact", model_metadata)
        
        response = client.get(
            "/api/v1/models/model_prod123_v1",
            headers=api_key_header
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all metadata fields
        assert data["model_id"] == "model_prod123_v1"
        assert data["product_id"] == "PROD-123"
        assert data["model_type"] == "custom"
        assert data["version"] == 1
        assert data["artifact_path"] == "s3://bucket/model1"
        assert data["training_dataset_id"] == "dataset_123"
        assert data["mae"] == 5.2
        assert data["rmse"] == 7.8
        assert data["mape"] == 4.5
        assert data["hyperparameters"] == {"n_estimators": 100}
        assert data["forecast_horizon"] == 30
        assert "created_at" in data
        
        # Verify registry was called correctly
        mock_registry.get_model.assert_called_once_with("model_prod123_v1")


def test_get_model_metadata_not_found(client, api_key_header):
    """
    Test getting model metadata for non-existent model.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.get_model.side_effect = ValueError("Model not found")
        
        response = client.get(
            "/api/v1/models/nonexistent_model",
            headers=api_key_header
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_model_metadata_no_auth(client):
    """
    Test getting model metadata without authentication.
    
    **Validates: Requirement 10.5**
    """
    response = client.get("/api/v1/models/model_prod123_v1")
    
    assert response.status_code == 401


def test_get_model_metadata_registry_error(client, api_key_header):
    """
    Test getting model metadata when registry fails.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.api.main.model_registry") as mock_registry:
        mock_registry.get_model.side_effect = RuntimeError("S3 error")
        
        response = client.get(
            "/api/v1/models/model_prod123_v1",
            headers=api_key_header
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]


# ============================================================================
# Tests for POST /api/v1/data/ingest
# ============================================================================


def test_ingest_data_success_list_format(client, api_key_header, mock_ingestion_result_success):
    """
    Test data ingestion with list of records.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.data.ingestion.DataIngestionService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.ingest_batch.return_value = mock_ingestion_result_success
        
        data = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "product_id": "PROD-123",
                "sales_volume": 100.5,
                "price": 19.99,
                "is_holiday": False,
                "day_of_week": 0,
                "month": 1,
                "quarter": 1
            },
            {
                "timestamp": "2025-01-02T00:00:00",
                "product_id": "PROD-123",
                "sales_volume": 105.2,
                "price": 19.99,
                "is_holiday": False,
                "day_of_week": 1,
                "month": 1,
                "quarter": 1
            }
        ]
        
        response = client.post(
            "/api/v1/data/ingest",
            json={
                "data": data,
                "format": "auto"
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert response_data["success"] is True
        assert response_data["record_count"] == 1000
        assert response_data["s3_path"] == "s3://bucket/historical_data/dataset_20250115_123456"
        assert response_data["errors"] == []
        assert response_data["ingestion_time_seconds"] == 2.5
        
        # Verify service was called correctly
        mock_service.ingest_batch.assert_called_once()
        call_args = mock_service.ingest_batch.call_args
        assert call_args[1]["data"] == data
        assert call_args[1]["format"] == "auto"


def test_ingest_data_success_csv_format(client, api_key_header, mock_ingestion_result_success):
    """
    Test data ingestion with CSV string.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.data.ingestion.DataIngestionService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.ingest_batch.return_value = mock_ingestion_result_success
        
        csv_data = """timestamp,product_id,sales_volume,price,is_holiday,day_of_week,month,quarter
2025-01-01T00:00:00,PROD-123,100.5,19.99,False,0,1,1
2025-01-02T00:00:00,PROD-123,105.2,19.99,False,1,1,1"""
        
        response = client.post(
            "/api/v1/data/ingest",
            json={
                "data": csv_data,
                "format": "csv"
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["record_count"] == 1000


def test_ingest_data_validation_failure(client, api_key_header, mock_ingestion_result_failure):
    """
    Test data ingestion with validation errors.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.data.ingestion.DataIngestionService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.ingest_batch.return_value = mock_ingestion_result_failure
        
        data = [
            {
                "product_id": "PROD-123",
                "sales_volume": 100.5
                # Missing required fields
            }
        ]
        
        response = client.post(
            "/api/v1/data/ingest",
            json={
                "data": data,
                "format": "auto"
            },
            headers=api_key_header
        )
        
        assert response.status_code == 400
        response_data = response.json()
        
        assert "detail" in response_data
        assert "error" in response_data["detail"]
        assert response_data["detail"]["error"]["code"] == "VALIDATION_ERROR"


def test_ingest_data_no_auth(client):
    """
    Test data ingestion without authentication.
    
    **Validates: Requirement 10.5**
    """
    response = client.post(
        "/api/v1/data/ingest",
        json={
            "data": [],
            "format": "auto"
        }
    )
    
    assert response.status_code == 401


def test_ingest_data_service_error(client, api_key_header):
    """
    Test data ingestion when service fails.
    
    **Validates: Requirement 10.5**
    """
    with patch("src.data.ingestion.DataIngestionService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.ingest_batch.side_effect = RuntimeError("S3 error")
        
        response = client.post(
            "/api/v1/data/ingest",
            json={
                "data": [{"timestamp": "2025-01-01", "product_id": "PROD-123"}],
                "format": "auto"
            },
            headers=api_key_header
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]


def test_ingest_data_missing_data_field(client, api_key_header):
    """
    Test data ingestion with missing data field.
    
    **Validates: Requirement 10.5**
    """
    response = client.post(
        "/api/v1/data/ingest",
        json={
            "format": "auto"
            # Missing 'data' field
        },
        headers=api_key_header
    )
    
    # Should return 400 (validation error handled by error handler)
    assert response.status_code == 400


def test_ingest_data_invalid_format(client, api_key_header):
    """
    Test data ingestion with invalid format parameter.
    
    **Validates: Requirement 10.5**
    """
    response = client.post(
        "/api/v1/data/ingest",
        json={
            "data": [],
            "format": "invalid"
        },
        headers=api_key_header
    )
    
    # Should return 400 (validation error handled by error handler)
    assert response.status_code == 400


def test_ingest_data_large_batch(client, api_key_header):
    """
    Test data ingestion with large batch.
    
    **Validates: Requirement 10.5**
    """
    large_result = IngestionResult(
        success=True,
        record_count=5_000_000,
        s3_path="s3://bucket/historical_data/dataset_large",
        validation_result=ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            record_count=5_000_000
        ),
        errors=[],
        ingestion_time_seconds=55.0
    )
    
    with patch("src.data.ingestion.DataIngestionService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.ingest_batch.return_value = large_result
        
        # Simulate large batch (just send metadata, not actual 5M records)
        response = client.post(
            "/api/v1/data/ingest",
            json={
                "data": [{"timestamp": "2025-01-01", "product_id": "PROD-123"}] * 100,
                "format": "auto"
            },
            headers=api_key_header
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["record_count"] == 5_000_000
        assert response_data["ingestion_time_seconds"] == 55.0
