"""Unit tests for Model Registry operations."""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
import json

from src.registry.model_registry import ModelRegistry, ModelMetadata


@pytest.fixture
def sample_metadata():
    """Create sample model metadata for testing."""
    return ModelMetadata(
        model_id="model_test_001",
        product_id="product_123",
        model_type="custom",
        version=1,
        artifact_path="s3://test-bucket/product_123/custom/v1/model_test_001",
        training_dataset_id="dataset_456",
        mae=10.5,
        rmse=15.2,
        mape=5.3,
        hyperparameters={"learning_rate": 0.01, "epochs": 100},
        created_at=datetime(2025, 1, 15, 10, 30, 0),
        forecast_horizon=30
    )


@pytest.fixture
def sample_model_artifact():
    """Create sample model artifact (serialized bytes)."""
    return b"mock_serialized_model_data"


@pytest.fixture
def mock_registry():
    """Create ModelRegistry with mocked dependencies."""
    with patch('src.registry.model_registry.boto3.client') as mock_boto3, \
         patch('src.registry.model_registry.db_manager') as mock_db:
        
        # Mock S3 client
        mock_s3 = Mock()
        mock_boto3.return_value = mock_s3
        
        # Mock database session
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None
        
        registry = ModelRegistry()
        registry.s3_client = mock_s3
        
        yield registry, mock_s3, mock_session


class TestModelRegistration:
    """Test model registration functionality."""
    
    def test_register_model_success(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test successful model registration."""
        registry, mock_s3, mock_session = mock_registry
        
        # Execute registration
        model_id = registry.register_model(sample_model_artifact, sample_metadata)
        
        # Verify S3 upload was called
        assert mock_s3.put_object.called
        call_args = mock_s3.put_object.call_args
        assert call_args[1]['Bucket'] == registry.bucket_name
        assert call_args[1]['Body'] == sample_model_artifact
        assert 'product_123' in call_args[1]['Key']
        assert 'custom' in call_args[1]['Key']
        
        # Verify database insert was called
        assert mock_session.execute.called
        
        # Verify return value
        assert model_id == sample_metadata.model_id
    
    def test_register_model_invalid_model_id(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration fails with empty model_id."""
        registry, _, _ = mock_registry
        sample_metadata.model_id = ""
        
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            registry.register_model(sample_model_artifact, sample_metadata)
    
    def test_register_model_invalid_product_id(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration fails with empty product_id."""
        registry, _, _ = mock_registry
        sample_metadata.product_id = ""
        
        with pytest.raises(ValueError, match="product_id cannot be empty"):
            registry.register_model(sample_model_artifact, sample_metadata)
    
    def test_register_model_invalid_model_type(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration fails with invalid model_type."""
        registry, _, _ = mock_registry
        sample_metadata.model_type = "invalid_type"
        
        with pytest.raises(ValueError, match="Invalid model_type"):
            registry.register_model(sample_model_artifact, sample_metadata)
    
    def test_register_model_invalid_version(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration fails with invalid version."""
        registry, _, _ = mock_registry
        sample_metadata.version = 0
        
        with pytest.raises(ValueError, match="Invalid version"):
            registry.register_model(sample_model_artifact, sample_metadata)
    
    def test_register_model_s3_error(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration handles S3 errors."""
        registry, mock_s3, _ = mock_registry
        
        # Mock S3 error
        mock_s3.put_object.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}},
            'PutObject'
        )
        
        with pytest.raises(RuntimeError, match="S3 error while registering model"):
            registry.register_model(sample_model_artifact, sample_metadata)
    
    def test_register_model_database_error(self, mock_registry, sample_metadata, sample_model_artifact):
        """Test registration handles database errors."""
        registry, mock_s3, mock_session = mock_registry
        
        # Mock database error
        mock_session.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(RuntimeError, match="Error registering model"):
            registry.register_model(sample_model_artifact, sample_metadata)


class TestModelRetrieval:
    """Test model retrieval functionality."""
    
    def test_get_model_success(self, mock_registry):
        """Test successful model retrieval."""
        registry, mock_s3, mock_session = mock_registry
        
        # Mock database query result
        mock_result = (
            "model_test_001",
            "product_123",
            "custom",
            1,
            "s3://test-bucket/product_123/custom/v1/model_test_001",
            "dataset_456",
            10.5,
            15.2,
            5.3,
            json.dumps({"learning_rate": 0.01}),
            datetime(2025, 1, 15, 10, 30, 0),
            30
        )
        mock_session.execute.return_value.fetchone.return_value = mock_result
        
        # Mock S3 response
        mock_s3.get_object.return_value = {
            'Body': Mock(read=lambda: b"mock_model_data")
        }
        
        # Execute retrieval
        artifact, metadata = registry.get_model("model_test_001")
        
        # Verify results
        assert artifact == b"mock_model_data"
        assert metadata.model_id == "model_test_001"
        assert metadata.product_id == "product_123"
        assert metadata.model_type == "custom"
        assert metadata.version == 1
        assert metadata.mae == 10.5
        assert metadata.hyperparameters == {"learning_rate": 0.01}
    
    def test_get_model_not_found(self, mock_registry):
        """Test retrieval fails when model not found."""
        registry, _, mock_session = mock_registry
        
        # Mock empty database result
        mock_session.execute.return_value.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="Model not found"):
            registry.get_model("nonexistent_model")
    
    def test_get_model_invalid_s3_path(self, mock_registry):
        """Test retrieval fails with invalid S3 path."""
        registry, _, mock_session = mock_registry
        
        # Mock database result with invalid S3 path
        mock_result = (
            "model_test_001", "product_123", "custom", 1,
            "invalid_path",  # Invalid S3 path
            "dataset_456", 10.5, 15.2, 5.3,
            json.dumps({}), datetime.now(), 30
        )
        mock_session.execute.return_value.fetchone.return_value = mock_result
        
        with pytest.raises(ValueError, match="Invalid S3 path"):
            registry.get_model("model_test_001")
    
    def test_get_model_s3_error(self, mock_registry):
        """Test retrieval handles S3 errors."""
        registry, mock_s3, mock_session = mock_registry
        
        # Mock database result
        mock_result = (
            "model_test_001", "product_123", "custom", 1,
            "s3://test-bucket/key",
            "dataset_456", 10.5, 15.2, 5.3,
            json.dumps({}), datetime.now(), 30
        )
        mock_session.execute.return_value.fetchone.return_value = mock_result
        
        # Mock S3 error
        mock_s3.get_object.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}},
            'GetObject'
        )
        
        with pytest.raises(RuntimeError, match="S3 error while retrieving model"):
            registry.get_model("model_test_001")


class TestModelListing:
    """Test model listing functionality."""
    
    def test_list_models_no_filter(self, mock_registry):
        """Test listing all models without filters."""
        registry, _, mock_session = mock_registry
        
        # Mock database results
        mock_results = [
            ("model_001", "product_123", "custom", 1, "s3://bucket/path1",
             "dataset_1", 10.5, 15.2, 5.3, json.dumps({}), datetime.now(), 30),
            ("model_002", "product_456", "forecast", 1, "s3://bucket/path2",
             "dataset_2", 12.3, 18.1, 6.2, json.dumps({}), datetime.now(), 30),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results
        
        # Execute listing
        models = registry.list_models()
        
        # Verify results
        assert len(models) == 2
        assert models[0].model_id == "model_001"
        assert models[1].model_id == "model_002"
    
    def test_list_models_filter_by_product(self, mock_registry):
        """Test listing models filtered by product_id."""
        registry, _, mock_session = mock_registry
        
        # Mock database results
        mock_results = [
            ("model_001", "product_123", "custom", 1, "s3://bucket/path1",
             "dataset_1", 10.5, 15.2, 5.3, json.dumps({}), datetime.now(), 30),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results
        
        # Execute listing with filter
        models = registry.list_models(product_id="product_123")
        
        # Verify query was called with filter
        call_args = mock_session.execute.call_args
        assert 'product_id' in call_args[0][1]
        assert len(models) == 1
        assert models[0].product_id == "product_123"
    
    def test_list_models_filter_by_type(self, mock_registry):
        """Test listing models filtered by model_type."""
        registry, _, mock_session = mock_registry
        
        # Mock database results
        mock_results = [
            ("model_001", "product_123", "custom", 1, "s3://bucket/path1",
             "dataset_1", 10.5, 15.2, 5.3, json.dumps({}), datetime.now(), 30),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results
        
        # Execute listing with filter
        models = registry.list_models(model_type="custom")
        
        # Verify query was called with filter
        call_args = mock_session.execute.call_args
        assert 'model_type' in call_args[0][1]
        assert len(models) == 1
        assert models[0].model_type == "custom"
    
    def test_list_models_empty_result(self, mock_registry):
        """Test listing returns empty list when no models found."""
        registry, _, mock_session = mock_registry
        
        # Mock empty database result
        mock_session.execute.return_value.fetchall.return_value = []
        
        # Execute listing
        models = registry.list_models()
        
        # Verify empty list
        assert len(models) == 0


class TestLatestModelRetrieval:
    """Test latest model retrieval functionality."""
    
    def test_get_latest_model_success(self, mock_registry):
        """Test successful retrieval of latest model."""
        registry, _, mock_session = mock_registry
        
        # Mock database result
        mock_result = (
            "model_test_003", "product_123", "custom", 3,
            "s3://bucket/path", "dataset_456",
            10.5, 15.2, 5.3, json.dumps({}), datetime.now(), 30
        )
        mock_session.execute.return_value.fetchone.return_value = mock_result
        
        # Execute retrieval
        model_id, metadata = registry.get_latest_model("product_123", "custom")
        
        # Verify results
        assert model_id == "model_test_003"
        assert metadata.version == 3
        assert metadata.product_id == "product_123"
        assert metadata.model_type == "custom"
    
    def test_get_latest_model_not_found(self, mock_registry):
        """Test retrieval fails when no model found."""
        registry, _, mock_session = mock_registry
        
        # Mock empty database result
        mock_session.execute.return_value.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="No model found"):
            registry.get_latest_model("nonexistent_product", "custom")
    
    def test_get_latest_model_database_error(self, mock_registry):
        """Test retrieval handles database errors."""
        registry, _, mock_session = mock_registry
        
        # Mock database error
        mock_session.execute.side_effect = Exception("Database error")
        
        with pytest.raises(RuntimeError, match="Error getting latest model"):
            registry.get_latest_model("product_123", "custom")


class TestVersionManagement:
    """Test model version management."""
    
    def test_multiple_versions_per_model(self, mock_registry, sample_model_artifact):
        """Test registering multiple versions of the same model."""
        registry, mock_s3, mock_session = mock_registry
        
        # Register version 1
        metadata_v1 = ModelMetadata(
            model_id="model_v1", product_id="product_123", model_type="custom",
            version=1, artifact_path="", training_dataset_id="dataset_1",
            mae=10.5, rmse=15.2, mape=5.3, hyperparameters={},
            created_at=datetime.now(), forecast_horizon=30
        )
        registry.register_model(sample_model_artifact, metadata_v1)
        
        # Register version 2
        metadata_v2 = ModelMetadata(
            model_id="model_v2", product_id="product_123", model_type="custom",
            version=2, artifact_path="", training_dataset_id="dataset_1",
            mae=9.2, rmse=13.8, mape=4.7, hyperparameters={},
            created_at=datetime.now(), forecast_horizon=30
        )
        registry.register_model(sample_model_artifact, metadata_v2)
        
        # Verify both registrations succeeded
        assert mock_s3.put_object.call_count == 2
        assert mock_session.execute.call_count == 2
    
    def test_version_100_plus_support(self, mock_registry):
        """Test support for 100+ versions per model (Requirement 7.5)."""
        registry, _, mock_session = mock_registry
        
        # Mock database results with 100+ versions
        mock_results = [
            (f"model_{i:03d}", "product_123", "custom", i, "s3://bucket/path",
             "dataset_1", 10.5, 15.2, 5.3, json.dumps({}), datetime.now(), 30)
            for i in range(1, 101)
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results
        
        # Execute listing
        models = registry.list_models(product_id="product_123")
        
        # Verify all 100 versions are returned
        assert len(models) == 100


class TestMetadataDataclass:
    """Test ModelMetadata dataclass functionality."""
    
    def test_metadata_to_dict(self, sample_metadata):
        """Test conversion of metadata to dictionary."""
        metadata_dict = sample_metadata.to_dict()
        
        assert metadata_dict['model_id'] == "model_test_001"
        assert metadata_dict['product_id'] == "product_123"
        assert metadata_dict['model_type'] == "custom"
        assert metadata_dict['version'] == 1
        assert metadata_dict['mae'] == 10.5
        assert isinstance(metadata_dict['created_at'], str)  # ISO format
        assert metadata_dict['hyperparameters'] == {"learning_rate": 0.01, "epochs": 100}
