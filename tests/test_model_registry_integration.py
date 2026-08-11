"""Integration tests for Model Registry with database."""
import pytest
from datetime import datetime
import joblib
from io import BytesIO

from src.registry.model_registry import ModelRegistry, ModelMetadata


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def registry():
    """Create ModelRegistry instance for integration testing."""
    return ModelRegistry()


@pytest.fixture
def sample_model_artifact():
    """Create a realistic serialized model artifact."""
    # Create a simple mock model (dictionary simulating a trained model)
    mock_model = {
        'model_type': 'linear_regression',
        'coefficients': [1.5, 2.3, -0.8],
        'intercept': 10.2
    }
    
    # Serialize using joblib
    buffer = BytesIO()
    joblib.dump(mock_model, buffer)
    return buffer.getvalue()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        model_id=f"test_model_{datetime.now().timestamp()}",
        product_id="test_product_001",
        model_type="custom",
        version=1,
        artifact_path="",  # Will be set by register_model
        training_dataset_id="test_dataset_001",
        mae=12.5,
        rmse=18.3,
        mape=6.2,
        hyperparameters={
            "algorithm": "linear_regression",
            "learning_rate": 0.01,
            "max_iterations": 1000
        },
        created_at=datetime.now(),
        forecast_horizon=30
    )


class TestModelRegistryIntegration:
    """Integration tests for Model Registry operations."""
    
    @pytest.mark.skip(reason="Requires live database and S3 connection")
    def test_register_and_retrieve_model(self, registry, sample_model_artifact, sample_metadata):
        """Test full cycle: register model, then retrieve it."""
        # Register model
        model_id = registry.register_model(sample_model_artifact, sample_metadata)
        assert model_id == sample_metadata.model_id
        
        # Retrieve model
        retrieved_artifact, retrieved_metadata = registry.get_model(model_id)
        
        # Verify artifact matches
        assert retrieved_artifact == sample_model_artifact
        
        # Verify metadata matches
        assert retrieved_metadata.model_id == sample_metadata.model_id
        assert retrieved_metadata.product_id == sample_metadata.product_id
        assert retrieved_metadata.model_type == sample_metadata.model_type
        assert retrieved_metadata.version == sample_metadata.version
        assert retrieved_metadata.mae == sample_metadata.mae
        assert retrieved_metadata.rmse == sample_metadata.rmse
        assert retrieved_metadata.mape == sample_metadata.mape
    
    @pytest.mark.skip(reason="Requires live database and S3 connection")
    def test_list_models_with_filters(self, registry, sample_model_artifact, sample_metadata):
        """Test listing models with various filters."""
        # Register a model
        registry.register_model(sample_model_artifact, sample_metadata)
        
        # List all models for this product
        models = registry.list_models(product_id=sample_metadata.product_id)
        assert len(models) > 0
        assert any(m.model_id == sample_metadata.model_id for m in models)
        
        # List models by type
        custom_models = registry.list_models(
            product_id=sample_metadata.product_id,
            model_type="custom"
        )
        assert len(custom_models) > 0
        assert all(m.model_type == "custom" for m in custom_models)
    
    @pytest.mark.skip(reason="Requires live database and S3 connection")
    def test_get_latest_model_version(self, registry, sample_model_artifact):
        """Test retrieving the latest version of a model."""
        product_id = "test_product_versioning"
        
        # Register multiple versions
        for version in [1, 2, 3]:
            metadata = ModelMetadata(
                model_id=f"test_model_v{version}_{datetime.now().timestamp()}",
                product_id=product_id,
                model_type="custom",
                version=version,
                artifact_path="",
                training_dataset_id="test_dataset_001",
                mae=10.0 - version,  # Improving metrics
                rmse=15.0 - version,
                mape=5.0 - version,
                hyperparameters={},
                created_at=datetime.now(),
                forecast_horizon=30
            )
            registry.register_model(sample_model_artifact, metadata)
        
        # Get latest version
        model_id, latest_metadata = registry.get_latest_model(product_id, "custom")
        
        # Verify it's version 3
        assert latest_metadata.version == 3
        assert latest_metadata.product_id == product_id
    
    @pytest.mark.skip(reason="Requires live database and S3 connection")
    def test_model_artifact_deserialization(self, registry, sample_model_artifact, sample_metadata):
        """Test that retrieved model artifacts can be deserialized."""
        # Register model
        model_id = registry.register_model(sample_model_artifact, sample_metadata)
        
        # Retrieve model
        retrieved_artifact, _ = registry.get_model(model_id)
        
        # Deserialize the artifact
        buffer = BytesIO(retrieved_artifact)
        model = joblib.load(buffer)
        
        # Verify model structure
        assert 'model_type' in model
        assert 'coefficients' in model
        assert model['model_type'] == 'linear_regression'


class TestDatabaseIndexing:
    """Test database indexing performance."""
    
    @pytest.mark.skip(reason="Requires live database connection")
    def test_product_type_index_performance(self, registry):
        """Verify (product_id, model_type) index improves query performance."""
        # This test would measure query performance with EXPLAIN ANALYZE
        # to verify the index is being used
        pass
    
    @pytest.mark.skip(reason="Requires live database connection")
    def test_created_at_index_performance(self, registry):
        """Verify created_at index improves sorting performance."""
        # This test would measure query performance for time-based queries
        pass


# Note: These integration tests are skipped by default and should be run
# manually when database and S3 are available. To run:
# pytest tests/test_model_registry_integration.py -v -m integration --run-integration
