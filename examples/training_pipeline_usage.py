"""
Example usage of the Training Pipeline orchestration.

This example demonstrates how to use the TrainingPipeline class to orchestrate
the complete training workflow from data loading to model registration.
"""

from src.training.pipeline import TrainingPipeline, PipelineConfig
from src.training.custom_model_trainer import ModelConfig


def main():
    """Example: Train a custom forecasting model using the training pipeline."""
    
    # Initialize the training pipeline
    pipeline = TrainingPipeline()
    
    # Configure the pipeline
    config = PipelineConfig(
        dataset_path="historical-data/product_123",  # S3 path to dataset
        product_id="product_123",
        model_config=ModelConfig(
            algorithm="random_forest",
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            features=[],  # Empty list means use all available features
            forecast_horizon=30
        ),
        train_split=0.8,
        target_column='sales_volume',
        training_dataset_id='dataset_v1_2024',
        max_retries=3,
        retry_base_delay=1.0
    )
    
    # Execute the training pipeline
    print("Starting training pipeline...")
    result = pipeline.train_custom_model(config)
    
    # Check results
    if result.success:
        print(f"\n✓ Training pipeline completed successfully!")
        print(f"  Model ID: {result.model_id}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"\n  Performance Metrics:")
        print(f"    MAE:  {result.metadata.mae:.2f}")
        print(f"    RMSE: {result.metadata.rmse:.2f}")
        print(f"    MAPE: {result.metadata.mape:.2f}%")
        print(f"\n  Stage Timings:")
        for stage, duration in result.stage_timings.items():
            print(f"    {stage}: {duration:.2f}s")
        
        if result.warnings:
            print(f"\n  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")
    else:
        print(f"\n✗ Training pipeline failed!")
        print(f"  Errors:")
        for error in result.errors:
            print(f"    - {error}")
        
        if result.warnings:
            print(f"\n  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")


def example_prophet_model():
    """Example: Train a Prophet model using the training pipeline."""
    
    pipeline = TrainingPipeline()
    
    config = PipelineConfig(
        dataset_path="historical-data/product_456",
        product_id="product_456",
        model_config=ModelConfig(
            algorithm="prophet",
            hyperparameters={
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05
            },
            features=['price', 'is_holiday'],  # Prophet will use these as regressors
            forecast_horizon=90
        ),
        train_split=0.8,
        target_column='sales_volume'
    )
    
    print("Training Prophet model...")
    result = pipeline.train_custom_model(config)
    
    if result.success:
        print(f"✓ Prophet model trained successfully: {result.model_id}")
    else:
        print(f"✗ Prophet model training failed: {result.errors}")


def example_with_retry_logic():
    """Example: Demonstrate retry logic for transient failures."""
    
    pipeline = TrainingPipeline()
    
    # Configure with aggressive retry settings
    config = PipelineConfig(
        dataset_path="historical-data/product_789",
        product_id="product_789",
        model_config=ModelConfig(
            algorithm="gradient_boosting",
            hyperparameters={
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5
            },
            forecast_horizon=60
        ),
        max_retries=5,  # More retries for unreliable network
        retry_base_delay=2.0  # Longer base delay
    )
    
    print("Training with retry logic enabled...")
    result = pipeline.train_custom_model(config)
    
    if result.success:
        print(f"✓ Model trained successfully after potential retries")
        print(f"  Total execution time: {result.execution_time:.2f}s")
    else:
        print(f"✗ Training failed even with retries: {result.errors}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run other examples:
    # example_prophet_model()
    # example_with_retry_logic()
