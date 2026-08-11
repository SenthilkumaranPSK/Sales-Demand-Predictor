"""
Example usage of TrainingPipeline for Amazon Forecast predictor training.

This example demonstrates how to use the TrainingPipeline class to train
Amazon Forecast predictors with AutoML or specific algorithms.

Requirements:
- AWS credentials configured
- IAM role ARN for Forecast to access S3
- Historical data in S3 or local file
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.training.pipeline import TrainingPipeline, ForecastPredictorConfig


def create_sample_data():
    """Create sample historical sales data."""
    # Generate 365 days of historical data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'product_id': ['prod_1'] * 365,
        'sales_volume': np.random.randint(50, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 7) * 20,  # Weekly seasonality
        'price': np.random.uniform(10, 50, 365),
        'is_holiday': [False] * 365,
        'day_of_week': [d.dayofweek for d in dates],
        'month': [d.month for d in dates],
        'quarter': [(d.month - 1) // 3 + 1 for d in dates]
    })
    
    # Mark some days as holidays
    holiday_indices = [0, 150, 200, 300, 364]  # New Year, Memorial Day, July 4th, Thanksgiving, New Year's Eve
    data.loc[holiday_indices, 'is_holiday'] = True
    
    return data


def example_1_automl_predictor():
    """
    Example 1: Train Forecast predictor with AutoML.
    
    This example demonstrates training a predictor using Amazon Forecast's
    AutoML feature, which automatically selects the best algorithm.
    """
    print("=" * 80)
    print("Example 1: Train Forecast Predictor with AutoML")
    print("=" * 80)
    
    # Create sample data and save to CSV
    data = create_sample_data()
    data.to_csv('sample_data.csv', index=False)
    print(f"Created sample data with {len(data)} records")
    
    # Initialize training pipeline
    # Note: In production, provide role_arn for Forecast to access S3
    pipeline = TrainingPipeline(
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'  # Replace with your IAM role ARN
    )
    
    # Configure Forecast predictor training
    config = ForecastPredictorConfig(
        dataset_path='sample_data.csv',  # Or S3 path: 's3://bucket/data.csv'
        product_id='prod_1',
        forecast_horizon=30,  # Forecast 30 days ahead
        dataset_name='sales-dataset',
        dataset_group_name='sales-dataset-group',
        predictor_name='sales-predictor-automl',
        algorithm='auto',  # Use AutoML
        dataset_frequency='D',  # Daily data
        timestamp_format='yyyy-MM-dd HH:mm:ss',
        training_dataset_id='dataset_v1',
        max_retries=3,
        retry_base_delay=1.0,
        max_wait_seconds=7200,  # Wait up to 2 hours for training
        poll_interval=60  # Check status every 60 seconds
    )
    
    # Train predictor
    print("\nStarting Forecast predictor training with AutoML...")
    result = pipeline.train_forecast_predictor(config)
    
    # Check results
    if result.success:
        print("\n✓ Predictor training completed successfully!")
        print(f"  Predictor ARN: {result.predictor_arn}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        
        if result.metrics:
            print("\n  Performance Metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"    {metric_name.upper()}: {metric_value:.4f}")
        
        print("\n  Stage Timings:")
        for stage, timing in result.stage_timings.items():
            print(f"    {stage}: {timing:.2f}s")
    else:
        print("\n✗ Predictor training failed!")
        print(f"  Errors: {result.errors}")
    
    if result.warnings:
        print("\n  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")


def example_2_specific_algorithm():
    """
    Example 2: Train Forecast predictor with specific algorithm.
    
    This example demonstrates training a predictor using a specific
    algorithm (ARIMA) instead of AutoML.
    """
    print("\n" + "=" * 80)
    print("Example 2: Train Forecast Predictor with Specific Algorithm (ARIMA)")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data()
    data.to_csv('sample_data.csv', index=False)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'
    )
    
    # Configure Forecast predictor with ARIMA algorithm
    config = ForecastPredictorConfig(
        dataset_path='sample_data.csv',
        product_id='prod_1',
        forecast_horizon=30,
        dataset_name='sales-dataset',
        dataset_group_name='sales-dataset-group',
        predictor_name='sales-predictor-arima',
        algorithm='arn:aws:forecast:::algorithm/ARIMA',  # Specific algorithm ARN
        dataset_frequency='D',
        timestamp_format='yyyy-MM-dd HH:mm:ss',
        training_dataset_id='dataset_v1'
    )
    
    # Train predictor
    print("\nStarting Forecast predictor training with ARIMA...")
    result = pipeline.train_forecast_predictor(config)
    
    # Check results
    if result.success:
        print("\n✓ Predictor training completed successfully!")
        print(f"  Predictor ARN: {result.predictor_arn}")
        print(f"  Algorithm: ARIMA")
        
        if result.metrics:
            print("\n  Performance Metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"    {metric_name.upper()}: {metric_value:.4f}")
    else:
        print("\n✗ Predictor training failed!")
        print(f"  Errors: {result.errors}")


def example_3_error_handling():
    """
    Example 3: Error handling and retry logic.
    
    This example demonstrates how the pipeline handles errors and
    implements retry logic for transient failures.
    """
    print("\n" + "=" * 80)
    print("Example 3: Error Handling and Retry Logic")
    print("=" * 80)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'
    )
    
    # Configure with invalid dataset path to trigger error
    config = ForecastPredictorConfig(
        dataset_path='nonexistent_file.csv',  # Invalid path
        product_id='prod_1',
        forecast_horizon=30,
        dataset_name='sales-dataset',
        dataset_group_name='sales-dataset-group',
        predictor_name='sales-predictor-error',
        algorithm='auto',
        max_retries=3,  # Will retry transient errors up to 3 times
        retry_base_delay=1.0  # Start with 1 second delay
    )
    
    # Train predictor (will fail due to invalid path)
    print("\nAttempting to train predictor with invalid dataset path...")
    result = pipeline.train_forecast_predictor(config)
    
    # Check results
    if not result.success:
        print("\n✓ Error handling working correctly!")
        print(f"  Errors detected: {len(result.errors)}")
        for i, error in enumerate(result.errors, 1):
            print(f"    {i}. {error}")
        
        print("\n  The pipeline correctly:")
        print("    - Detected the error")
        print("    - Attempted retries for transient errors")
        print("    - Returned detailed error information")


def example_4_compare_algorithms():
    """
    Example 4: Compare multiple algorithms.
    
    This example demonstrates training multiple predictors with different
    algorithms and comparing their performance.
    """
    print("\n" + "=" * 80)
    print("Example 4: Compare Multiple Algorithms")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data()
    data.to_csv('sample_data.csv', index=False)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'
    )
    
    # Define algorithms to compare
    algorithms = [
        ('AutoML', 'auto'),
        ('ARIMA', 'arn:aws:forecast:::algorithm/ARIMA'),
        ('ETS', 'arn:aws:forecast:::algorithm/ETS'),
        ('Prophet', 'arn:aws:forecast:::algorithm/Prophet')
    ]
    
    results = {}
    
    # Train predictor for each algorithm
    for algo_name, algo_arn in algorithms:
        print(f"\nTraining predictor with {algo_name}...")
        
        config = ForecastPredictorConfig(
            dataset_path='sample_data.csv',
            product_id='prod_1',
            forecast_horizon=30,
            dataset_name='sales-dataset',
            dataset_group_name='sales-dataset-group',
            predictor_name=f'sales-predictor-{algo_name.lower()}',
            algorithm=algo_arn,
            dataset_frequency='D',
            timestamp_format='yyyy-MM-dd HH:mm:ss',
            training_dataset_id='dataset_v1'
        )
        
        result = pipeline.train_forecast_predictor(config)
        results[algo_name] = result
    
    # Compare results
    print("\n" + "=" * 80)
    print("Algorithm Comparison Results")
    print("=" * 80)
    
    print(f"\n{'Algorithm':<15} {'Status':<10} {'RMSE':<10} {'Execution Time':<15}")
    print("-" * 50)
    
    for algo_name, result in results.items():
        status = "✓ Success" if result.success else "✗ Failed"
        rmse = result.metrics.get('rmse', 'N/A') if result.metrics else 'N/A'
        rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
        exec_time = f"{result.execution_time:.2f}s"
        
        print(f"{algo_name:<15} {status:<10} {rmse_str:<10} {exec_time:<15}")
    
    # Find best algorithm
    successful_results = {
        name: result for name, result in results.items()
        if result.success and result.metrics and 'rmse' in result.metrics
    }
    
    if successful_results:
        best_algo = min(
            successful_results.items(),
            key=lambda x: x[1].metrics['rmse']
        )
        
        print(f"\n✓ Best Algorithm: {best_algo[0]}")
        print(f"  RMSE: {best_algo[1].metrics['rmse']:.4f}")
        print(f"  Predictor ARN: {best_algo[1].predictor_arn}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Amazon Forecast Predictor Training Examples")
    print("=" * 80)
    
    try:
        # Example 1: AutoML predictor
        example_1_automl_predictor()
        
        # Example 2: Specific algorithm
        # example_2_specific_algorithm()
        
        # Example 3: Error handling
        # example_3_error_handling()
        
        # Example 4: Compare algorithms
        # example_4_compare_algorithms()
        
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
