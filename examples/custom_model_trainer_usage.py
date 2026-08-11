"""
Example usage of the Custom Model Trainer module.

This example demonstrates how to train custom forecasting models using
Prophet or scikit-learn algorithms, configure hyperparameters, serialize
models, and compute performance metrics.
"""

from src.training.custom_model_trainer import (
    CustomModelTrainer,
    ModelConfig,
    TrainingResult
)
from src.training.data_preparation import TrainingDataPreparation
from src.registry.model_registry import ModelRegistry, ModelMetadata
import uuid
from datetime import datetime


def example_train_random_forest():
    """Example: Train a Random Forest model."""
    print("=" * 60)
    print("Example: Train Random Forest Model")
    print("=" * 60)
    
    # Step 1: Prepare training data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    print(f"✓ Data prepared: {len(dataset.train_data)} train, {len(dataset.validation_data)} validation")
    
    # Step 2: Configure model
    config = ModelConfig(
        algorithm='random_forest',
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        },
        forecast_horizon=30
    )
    
    # Step 3: Train model
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if result.success:
        print(f"\n✓ Model trained successfully!")
        print(f"  Algorithm: {result.model_type}")
        print(f"  MAE: {result.metrics.mae:.2f}")
        print(f"  RMSE: {result.metrics.rmse:.2f}")
        print(f"  MAPE: {result.metrics.mape:.2f}%")
        print(f"  Validation samples: {result.metrics.sample_size}")
        print(f"  Model artifact size: {len(result.model_artifact)} bytes")
    else:
        print(f"\n✗ Training failed: {result.errors}")


def example_train_prophet():
    """Example: Train a Prophet model."""
    print("\n" + "=" * 60)
    print("Example: Train Prophet Model")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    
    # Configure Prophet model
    config = ModelConfig(
        algorithm='prophet',
        hyperparameters={
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05
        },
        features=['price', 'is_holiday'],  # Additional regressors
        forecast_horizon=30
    )
    
    # Train model
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if result.success:
        print(f"\n✓ Prophet model trained successfully!")
        print(f"  MAE: {result.metrics.mae:.2f}")
        print(f"  RMSE: {result.metrics.rmse:.2f}")
        print(f"  MAPE: {result.metrics.mape:.2f}%")
        print(f"  Features used: {result.metadata['features']}")
    else:
        print(f"\n✗ Training failed: {result.errors}")


def example_train_gradient_boosting():
    """Example: Train a Gradient Boosting model."""
    print("\n" + "=" * 60)
    print("Example: Train Gradient Boosting Model")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    
    # Configure Gradient Boosting model
    config = ModelConfig(
        algorithm='gradient_boosting',
        hyperparameters={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'random_state': 42
        },
        forecast_horizon=30
    )
    
    # Train model
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if result.success:
        print(f"\n✓ Gradient Boosting model trained successfully!")
        print(f"  MAE: {result.metrics.mae:.2f}")
        print(f"  RMSE: {result.metrics.rmse:.2f}")
        print(f"  MAPE: {result.metrics.mape:.2f}%")
    else:
        print(f"\n✗ Training failed: {result.errors}")


def example_custom_feature_selection():
    """Example: Train model with custom feature selection."""
    print("\n" + "=" * 60)
    print("Example: Custom Feature Selection")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    
    print(f"\nAvailable features: {dataset.feature_columns}")
    
    # Train with only price and holiday features
    config = ModelConfig(
        algorithm='random_forest',
        hyperparameters={'n_estimators': 50, 'random_state': 42},
        features=['price', 'is_holiday'],  # Custom feature subset
        forecast_horizon=30
    )
    
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if result.success:
        print(f"\n✓ Model trained with custom features!")
        print(f"  Features used: {result.metadata['features']}")
        print(f"  MAE: {result.metrics.mae:.2f}")
        print(f"  RMSE: {result.metrics.rmse:.2f}")
        print(f"  MAPE: {result.metrics.mape:.2f}%")
    else:
        print(f"\n✗ Training failed: {result.errors}")


def example_register_trained_model():
    """Example: Train model and register in Model Registry."""
    print("\n" + "=" * 60)
    print("Example: Train and Register Model")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    
    # Train model
    config = ModelConfig(
        algorithm='random_forest',
        hyperparameters={'n_estimators': 100, 'random_state': 42},
        forecast_horizon=30
    )
    
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if not result.success:
        print(f"✗ Training failed: {result.errors}")
        return
    
    print(f"✓ Model trained successfully!")
    
    # Register model in Model Registry
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    
    metadata = ModelMetadata(
        model_id=model_id,
        product_id='PROD_001',
        model_type='custom',
        version=1,
        artifact_path='',  # Will be set by registry
        training_dataset_id='dataset_001',
        mae=result.metrics.mae,
        rmse=result.metrics.rmse,
        mape=result.metrics.mape,
        hyperparameters=result.metadata['hyperparameters'],
        created_at=datetime.utcnow(),
        forecast_horizon=config.forecast_horizon
    )
    
    registry = ModelRegistry()
    registered_id = registry.register_model(
        model_artifact=result.model_artifact,
        metadata=metadata
    )
    
    print(f"\n✓ Model registered in Model Registry!")
    print(f"  Model ID: {registered_id}")
    print(f"  Product ID: {metadata.product_id}")
    print(f"  Version: {metadata.version}")
    print(f"  MAE: {metadata.mae:.2f}")
    print(f"  RMSE: {metadata.rmse:.2f}")
    print(f"  MAPE: {metadata.mape:.2f}%")


def example_model_serialization():
    """Example: Serialize and deserialize model."""
    print("\n" + "=" * 60)
    print("Example: Model Serialization/Deserialization")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    
    # Train model
    config = ModelConfig(
        algorithm='random_forest',
        hyperparameters={'n_estimators': 50, 'random_state': 42},
        forecast_horizon=30
    )
    
    trainer = CustomModelTrainer()
    result = trainer.train_model(dataset, config)
    
    if not result.success:
        print(f"✗ Training failed: {result.errors}")
        return
    
    print(f"✓ Model trained successfully!")
    print(f"  Model artifact size: {len(result.model_artifact)} bytes")
    
    # Deserialize model
    model, algorithm = CustomModelTrainer.deserialize_model(result.model_artifact)
    
    print(f"\n✓ Model deserialized successfully!")
    print(f"  Algorithm: {algorithm}")
    print(f"  Model type: {type(model).__name__}")
    
    # Test prediction with deserialized model
    X_test = dataset.validation_data[dataset.feature_columns].values[:5]
    predictions = model.predict(X_test)
    
    print(f"\n✓ Predictions from deserialized model:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred:.2f}")


def example_compare_algorithms():
    """Example: Compare different algorithms."""
    print("\n" + "=" * 60)
    print("Example: Compare Different Algorithms")
    print("=" * 60)
    
    # Prepare data
    data_prep = TrainingDataPreparation()
    prep_result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if not prep_result.success:
        print(f"✗ Data preparation failed: {prep_result.errors}")
        return
    
    dataset = prep_result.dataset
    trainer = CustomModelTrainer()
    
    # Train multiple algorithms
    algorithms = [
        ('linear', {}),
        ('ridge', {'alpha': 1.0}),
        ('random_forest', {'n_estimators': 50, 'random_state': 42}),
        ('gradient_boosting', {'n_estimators': 50, 'random_state': 42})
    ]
    
    results = []
    
    for algo, hyperparams in algorithms:
        config = ModelConfig(
            algorithm=algo,
            hyperparameters=hyperparams,
            forecast_horizon=30
        )
        
        result = trainer.train_model(dataset, config)
        
        if result.success:
            results.append({
                'algorithm': algo,
                'mae': result.metrics.mae,
                'rmse': result.metrics.rmse,
                'mape': result.metrics.mape
            })
    
    # Display comparison
    print("\n✓ Algorithm Comparison:")
    print(f"\n{'Algorithm':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['algorithm']:<20} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['mape']:<10.2f}%")
    
    # Find best algorithm
    best = min(results, key=lambda x: x['mae'])
    print(f"\n✓ Best algorithm (by MAE): {best['algorithm']}")


def example_error_handling():
    """Example: Error handling scenarios."""
    print("\n" + "=" * 60)
    print("Example: Error Handling")
    print("=" * 60)
    
    trainer = CustomModelTrainer()
    
    # Example 1: Unsupported algorithm
    print("\n1. Unsupported algorithm:")
    from src.training.data_preparation import TrainingDataset
    import pandas as pd
    
    dummy_dataset = TrainingDataset(
        train_data=pd.DataFrame({'sales_volume': [1, 2, 3], 'price': [10, 20, 30]}),
        validation_data=pd.DataFrame({'sales_volume': [4, 5], 'price': [40, 50]}),
        feature_columns=['price'],
        target_column='sales_volume',
        normalization_params={},
        metadata={}
    )
    
    config = ModelConfig(
        algorithm='unsupported_algo',  # type: ignore
        hyperparameters={},
        forecast_horizon=30
    )
    
    result = trainer.train_model(dummy_dataset, config)
    if not result.success:
        print(f"   ✓ Error caught: {result.errors[0]}")
    
    # Example 2: Invalid forecast horizon
    print("\n2. Invalid forecast horizon:")
    config = ModelConfig(
        algorithm='random_forest',
        hyperparameters={},
        forecast_horizon=0  # Invalid
    )
    
    result = trainer.train_model(dummy_dataset, config)
    if not result.success:
        print(f"   ✓ Error caught: {result.errors[0]}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Custom Model Trainer Usage Examples")
    print("=" * 60)
    
    # Note: These examples assume you have:
    # 1. AWS credentials configured
    # 2. Historical data stored in S3
    # 3. Model Registry database set up
    
    print("\nNote: These examples require actual data to run.")
    print("They demonstrate the API usage patterns.\n")
    
    # Uncomment to run examples (requires actual data):
    # example_train_random_forest()
    # example_train_prophet()
    # example_train_gradient_boosting()
    # example_custom_feature_selection()
    # example_register_trained_model()
    # example_model_serialization()
    # example_compare_algorithms()
    # example_error_handling()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
