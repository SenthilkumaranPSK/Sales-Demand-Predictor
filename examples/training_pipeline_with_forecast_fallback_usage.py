"""
Example usage of Training Pipeline with Forecast fallback logic.

This example demonstrates:
1. Training both custom and Forecast models with fallback
2. Handling Forecast failures gracefully
3. Registering Forecast predictors in Model Registry
4. Generating model comparison reports
"""

from src.training.pipeline import (
    TrainingPipeline,
    PipelineConfig,
    ForecastPredictorConfig
)
from src.training.custom_model_trainer import ModelConfig


def example_train_with_forecast_fallback():
    """
    Example: Train custom model with Forecast benchmark and fallback logic.
    
    This demonstrates the complete workflow:
    - Train custom model (primary)
    - Train Forecast predictor (benchmark)
    - Register both models in Model Registry
    - Generate comparison report
    - Handle Forecast failures gracefully
    """
    print("=" * 80)
    print("Training Pipeline with Forecast Fallback Example")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'
    )
    
    # Configure custom model training
    custom_config = PipelineConfig(
        dataset_path='s3://demand-forecasting-historical-data/product_001/sales_history.csv',
        product_id='product_001',
        model_config=ModelConfig(
            algorithm='random_forest',
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            forecast_horizon=30
        ),
        train_split=0.8,
        target_column='sales_volume',
        training_dataset_id='dataset_v1',
        max_retries=3,
        retry_base_delay=1.0
    )
    
    # Configure Forecast predictor training
    forecast_config = ForecastPredictorConfig(
        dataset_path='s3://demand-forecasting-historical-data/product_001/sales_history.csv',
        product_id='product_001',
        forecast_horizon=30,
        dataset_name='product_001_dataset',
        dataset_group_name='product_001_group',
        predictor_name='product_001_predictor',
        algorithm='auto',  # Use AutoML
        dataset_frequency='D',
        timestamp_format='yyyy-MM-dd HH:mm:ss',
        training_dataset_id='dataset_v1',
        max_retries=3,
        retry_base_delay=1.0,
        max_wait_seconds=7200,  # 2 hours
        poll_interval=60  # Check every minute
    )
    
    # Train both models with fallback
    print("\n1. Training custom model and Forecast predictor...")
    results = pipeline.train_with_forecast_fallback(
        custom_config=custom_config,
        forecast_config=forecast_config
    )
    
    # Display custom model results
    print("\n2. Custom Model Results:")
    print("-" * 80)
    custom_result = results['custom_result']
    
    if custom_result.success:
        print(f"✓ Custom model training succeeded")
        print(f"  Model ID: {custom_result.model_id}")
        print(f"  MAE: {custom_result.metadata.mae:.2f}")
        print(f"  RMSE: {custom_result.metadata.rmse:.2f}")
        print(f"  MAPE: {custom_result.metadata.mape:.2f}%")
        print(f"  Execution time: {custom_result.execution_time:.2f}s")
    else:
        print(f"✗ Custom model training failed")
        print(f"  Errors: {custom_result.errors}")
    
    # Display Forecast predictor results
    print("\n3. Forecast Predictor Results:")
    print("-" * 80)
    forecast_result = results['forecast_result']
    
    if forecast_result:
        if forecast_result.success:
            print(f"✓ Forecast predictor training succeeded")
            print(f"  Predictor ARN: {forecast_result.predictor_arn}")
            print(f"  Model ID: {results['forecast_model_id']}")
            
            if forecast_result.metrics:
                print(f"  Metrics: {forecast_result.metrics}")
            
            print(f"  Execution time: {forecast_result.execution_time:.2f}s")
        else:
            print(f"✗ Forecast predictor training failed")
            print(f"  Errors: {forecast_result.errors}")
            print(f"  Note: Custom model training succeeded despite Forecast failure")
    else:
        print("  Forecast training was not attempted or failed early")
    
    # Display comparison report
    if results['comparison']:
        print("\n4. Model Comparison Report:")
        print("-" * 80)
        comparison = results['comparison']
        
        print(f"Custom Model:")
        print(f"  MAE: {comparison['custom']['mae']:.2f}")
        print(f"  RMSE: {comparison['custom']['rmse']:.2f}")
        print(f"  MAPE: {comparison['custom']['mape']:.2f}%")
        
        print(f"\nForecast Model:")
        print(f"  MAE: {comparison['forecast']['mae']:.2f}")
        print(f"  RMSE: {comparison['forecast']['rmse']:.2f}")
        print(f"  MAPE: {comparison['forecast']['mape']:.2f}%")
        
        print(f"\nImprovement (Custom vs Forecast):")
        print(f"  MAE: {comparison['improvement']['mae_pct']:.1f}%")
        print(f"  RMSE: {comparison['improvement']['rmse_pct']:.1f}%")
        print(f"  MAPE: {comparison['improvement']['mape_pct']:.1f}%")
        
        print(f"\nRecommendation: Use {comparison['recommendation']} model")
    else:
        print("\n4. Model Comparison:")
        print("-" * 80)
        print("  Comparison not available (Forecast training failed or incomplete)")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    return results


def example_register_forecast_predictor():
    """
    Example: Register an existing Forecast predictor in Model Registry.
    
    This is useful when you have a Forecast predictor that was trained
    outside the pipeline and want to track it in the Model Registry.
    """
    print("\n" + "=" * 80)
    print("Register Existing Forecast Predictor Example")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    # Register existing Forecast predictor
    predictor_arn = 'arn:aws:forecast:us-east-1:123456789012:predictor/my-predictor'
    
    print(f"\nRegistering Forecast predictor: {predictor_arn}")
    
    model_id = pipeline.register_forecast_predictor(
        predictor_arn=predictor_arn,
        product_id='product_001',
        forecast_horizon=30,
        metrics={
            'RMSE': 15.5,
            'wape': 0.08,
            'MASE': 1.2
        },
        training_dataset_id='dataset_v1',
        algorithm='auto'
    )
    
    if model_id:
        print(f"✓ Forecast predictor registered successfully")
        print(f"  Model ID: {model_id}")
    else:
        print(f"✗ Forecast predictor registration failed")
    
    print("=" * 80)


def example_custom_only_training():
    """
    Example: Train only custom model without Forecast benchmark.
    
    This demonstrates training without Forecast when:
    - Forecast is not needed for comparison
    - Forecast credentials are not available
    - Cost optimization is desired
    """
    print("\n" + "=" * 80)
    print("Custom Model Only Training Example")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    # Configure custom model training
    custom_config = PipelineConfig(
        dataset_path='s3://demand-forecasting-historical-data/product_002/sales_history.csv',
        product_id='product_002',
        model_config=ModelConfig(
            algorithm='xgboost',
            hyperparameters={
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            },
            forecast_horizon=30
        ),
        train_split=0.8,
        target_column='sales_volume',
        training_dataset_id='dataset_v2'
    )
    
    # Train custom model only (no Forecast config)
    print("\nTraining custom model only...")
    results = pipeline.train_with_forecast_fallback(
        custom_config=custom_config,
        forecast_config=None  # No Forecast training
    )
    
    # Display results
    custom_result = results['custom_result']
    
    if custom_result.success:
        print(f"\n✓ Custom model training succeeded")
        print(f"  Model ID: {custom_result.model_id}")
        print(f"  MAE: {custom_result.metadata.mae:.2f}")
        print(f"  RMSE: {custom_result.metadata.rmse:.2f}")
        print(f"  MAPE: {custom_result.metadata.mape:.2f}%")
    else:
        print(f"\n✗ Custom model training failed")
        print(f"  Errors: {custom_result.errors}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Example 1: Train with Forecast fallback
    try:
        example_train_with_forecast_fallback()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    # Example 2: Register existing Forecast predictor
    try:
        example_register_forecast_predictor()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    # Example 3: Train custom model only
    try:
        example_custom_only_training()
    except Exception as e:
        print(f"Error in example 3: {e}")
