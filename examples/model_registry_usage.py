"""
Example usage of the Model Registry.

This script demonstrates how to:
1. Register a trained model with metadata
2. Retrieve a model by ID
3. List available models with filters
4. Get the latest version of a model
"""
import joblib
from datetime import datetime
from io import BytesIO

from src.registry import model_registry, ModelMetadata


def example_register_model():
    """Example: Register a trained model."""
    print("=== Registering a Model ===")
    
    # 1. Train your model (example with a simple dict)
    trained_model = {
        'model_type': 'prophet',
        'parameters': {'seasonality_mode': 'multiplicative'},
        'trained_at': datetime.now().isoformat()
    }
    
    # 2. Serialize the model
    buffer = BytesIO()
    joblib.dump(trained_model, buffer)
    model_artifact = buffer.getvalue()
    
    # 3. Create metadata
    metadata = ModelMetadata(
        model_id="forecast_model_product123_v1",
        product_id="product_123",
        model_type="custom",
        version=1,
        artifact_path="",  # Will be set automatically
        training_dataset_id="dataset_2025_01_15",
        mae=12.5,
        rmse=18.3,
        mape=6.2,
        hyperparameters={
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05
        },
        created_at=datetime.now(),
        forecast_horizon=30
    )
    
    # 4. Register the model
    model_id = model_registry.register_model(model_artifact, metadata)
    print(f"Model registered successfully: {model_id}")
    print(f"S3 path: {metadata.artifact_path}")
    
    return model_id


def example_retrieve_model(model_id: str):
    """Example: Retrieve a model by ID."""
    print(f"\n=== Retrieving Model: {model_id} ===")
    
    # Retrieve model artifact and metadata
    artifact, metadata = model_registry.get_model(model_id)
    
    # Deserialize the model
    buffer = BytesIO(artifact)
    model = joblib.load(buffer)
    
    print(f"Model type: {metadata.model_type}")
    print(f"Product ID: {metadata.product_id}")
    print(f"Version: {metadata.version}")
    print(f"Performance metrics:")
    print(f"  - MAE: {metadata.mae}")
    print(f"  - RMSE: {metadata.rmse}")
    print(f"  - MAPE: {metadata.mape}")
    print(f"Hyperparameters: {metadata.hyperparameters}")
    
    return model, metadata


def example_list_models():
    """Example: List available models with filters."""
    print("\n=== Listing Models ===")
    
    # List all models for a specific product
    product_models = model_registry.list_models(product_id="product_123")
    print(f"\nModels for product_123: {len(product_models)}")
    for model in product_models:
        print(f"  - {model.model_id} (v{model.version}, MAE: {model.mae})")
    
    # List only custom models
    custom_models = model_registry.list_models(model_type="custom")
    print(f"\nCustom models: {len(custom_models)}")
    
    # List all models
    all_models = model_registry.list_models()
    print(f"\nTotal models: {len(all_models)}")


def example_get_latest_model():
    """Example: Get the latest version of a model."""
    print("\n=== Getting Latest Model ===")
    
    try:
        model_id, metadata = model_registry.get_latest_model(
            product_id="product_123",
            model_type="custom"
        )
        
        print(f"Latest model: {model_id}")
        print(f"Version: {metadata.version}")
        print(f"Created at: {metadata.created_at}")
        print(f"Performance: MAE={metadata.mae}, RMSE={metadata.rmse}, MAPE={metadata.mape}")
        
        return model_id, metadata
        
    except ValueError as e:
        print(f"No model found: {e}")
        return None, None


def example_version_management():
    """Example: Managing multiple versions of a model."""
    print("\n=== Version Management ===")
    
    product_id = "product_456"
    
    # Register multiple versions
    for version in [1, 2, 3]:
        trained_model = {
            'version': version,
            'improvements': f'Version {version} improvements'
        }
        
        buffer = BytesIO()
        joblib.dump(trained_model, buffer)
        model_artifact = buffer.getvalue()
        
        metadata = ModelMetadata(
            model_id=f"forecast_model_{product_id}_v{version}",
            product_id=product_id,
            model_type="custom",
            version=version,
            artifact_path="",
            training_dataset_id=f"dataset_v{version}",
            mae=15.0 - version,  # Improving metrics
            rmse=20.0 - version,
            mape=7.0 - version,
            hyperparameters={"version": version},
            created_at=datetime.now(),
            forecast_horizon=30
        )
        
        model_id = model_registry.register_model(model_artifact, metadata)
        print(f"Registered version {version}: {model_id} (MAE: {metadata.mae})")
    
    # Get latest version
    latest_id, latest_metadata = model_registry.get_latest_model(product_id, "custom")
    print(f"\nLatest version: {latest_metadata.version} (MAE: {latest_metadata.mae})")


def example_error_handling():
    """Example: Error handling."""
    print("\n=== Error Handling ===")
    
    # Try to retrieve non-existent model
    try:
        model_registry.get_model("nonexistent_model_id")
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Try to get latest model for non-existent product
    try:
        model_registry.get_latest_model("nonexistent_product", "custom")
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Try to register model with invalid metadata
    try:
        invalid_metadata = ModelMetadata(
            model_id="",  # Invalid: empty model_id
            product_id="product_123",
            model_type="custom",
            version=1,
            artifact_path="",
            training_dataset_id="dataset_001",
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            hyperparameters={},
            created_at=datetime.now(),
            forecast_horizon=30
        )
        model_registry.register_model(b"artifact", invalid_metadata)
    except ValueError as e:
        print(f"Expected validation error: {e}")


def main():
    """Run all examples."""
    print("Model Registry Usage Examples")
    print("=" * 50)
    
    # Note: These examples require a running database and S3 connection
    # Uncomment to run with actual infrastructure
    
    # model_id = example_register_model()
    # example_retrieve_model(model_id)
    # example_list_models()
    # example_get_latest_model()
    # example_version_management()
    # example_error_handling()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Uncomment the function calls in main() to run with actual infrastructure.")


if __name__ == "__main__":
    main()
