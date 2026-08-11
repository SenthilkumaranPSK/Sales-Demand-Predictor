"""
Example usage of Amazon Forecast integration for dataset import.

This example demonstrates how to:
1. Load historical data
2. Configure Amazon Forecast dataset
3. Import data into Amazon Forecast
4. Wait for import completion
"""

import pandas as pd
from datetime import datetime, timedelta

from src.training.forecast_integration import (
    AmazonForecastIntegration,
    ForecastDatasetConfig
)


def create_sample_data():
    """Create sample historical sales data."""
    # Generate 365 days of historical data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'product_id': ['PROD_001'] * 365,
        'sales_volume': [100 + i * 0.5 + (i % 7) * 10 for i in range(365)],
        'price': [19.99 + (i % 30) * 0.1 for i in range(365)],
        'is_holiday': [i % 7 == 0 or i % 7 == 6 for i in range(365)],
        'day_of_week': [i % 7 for i in range(365)],
        'month': [dates[i].month for i in range(365)],
        'quarter': [dates[i].quarter for i in range(365)]
    })
    
    return data


def main():
    """Main example workflow."""
    print("Amazon Forecast Integration Example")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n1. Creating sample historical data...")
    data = create_sample_data()
    print(f"   Created {len(data)} records for product PROD_001")
    print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Step 2: Configure Forecast dataset
    print("\n2. Configuring Forecast dataset...")
    config = ForecastDatasetConfig(
        dataset_name='demand_forecast_dataset',
        dataset_group_name='demand_forecast_group',
        domain='CUSTOM',
        dataset_frequency='D',  # Daily frequency
        timestamp_format='yyyy-MM-dd HH:mm:ss'
    )
    print(f"   Dataset name: {config.dataset_name}")
    print(f"   Dataset group: {config.dataset_group_name}")
    print(f"   Domain: {config.domain}")
    print(f"   Frequency: {config.dataset_frequency}")
    
    # Step 3: Initialize Forecast integration
    print("\n3. Initializing Amazon Forecast integration...")
    
    # NOTE: In production, provide a valid IAM role ARN that allows
    # Amazon Forecast to access your S3 bucket
    role_arn = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/ForecastRole'
    
    forecast_integration = AmazonForecastIntegration(
        role_arn=role_arn
    )
    print("   Integration initialized")
    
    # Step 4: Import dataset
    print("\n4. Importing dataset to Amazon Forecast...")
    print("   This will:")
    print("   - Convert data to Forecast format")
    print("   - Upload CSV to S3")
    print("   - Create dataset group and dataset")
    print("   - Start import job")
    
    result = forecast_integration.import_dataset(
        data=data,
        config=config,
        product_id='PROD_001'
    )
    
    if result.success:
        print("\n   ✓ Import initiated successfully!")
        print(f"   Dataset Group ARN: {result.dataset_group_arn}")
        print(f"   Dataset ARN: {result.dataset_arn}")
        print(f"   Import Job ARN: {result.import_job_arn}")
        print(f"   S3 Path: {result.s3_path}")
        print(f"   Records imported: {result.record_count}")
        
        if result.warnings:
            print("\n   Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        # Step 5: Wait for import completion (optional)
        print("\n5. Waiting for import job to complete...")
        print("   This may take several minutes...")
        
        completed = forecast_integration.wait_for_import_completion(
            import_job_arn=result.import_job_arn,
            max_wait_seconds=3600,  # Wait up to 1 hour
            poll_interval=60  # Check every minute
        )
        
        if completed:
            print("\n   ✓ Import completed successfully!")
            print("   Dataset is ready for predictor training.")
        else:
            print("\n   ✗ Import failed or timed out.")
            print("   Check AWS Forecast console for details.")
    
    else:
        print("\n   ✗ Import failed!")
        print("   Errors:")
        for error in result.errors:
            print(f"   - {error}")
    
    print("\n" + "=" * 50)
    print("Example completed")


def example_with_minimal_features():
    """Example with minimal required features (no related features)."""
    print("\nExample: Import with minimal features")
    print("-" * 50)
    
    # Create data with only required columns
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'product_id': ['PROD_002'] * 100,
        'sales_volume': [50 + i * 0.3 for i in range(100)]
    })
    
    print(f"Created {len(data)} records with minimal features")
    print(f"Columns: {list(data.columns)}")
    
    config = ForecastDatasetConfig(
        dataset_name='minimal_dataset',
        dataset_group_name='minimal_group',
        domain='CUSTOM',
        dataset_frequency='D'
    )
    
    # Initialize and import
    forecast_integration = AmazonForecastIntegration(
        role_arn='arn:aws:iam::YOUR_ACCOUNT_ID:role/ForecastRole'
    )
    
    result = forecast_integration.import_dataset(
        data=data,
        config=config,
        product_id='PROD_002'
    )
    
    if result.success:
        print(f"✓ Import successful: {result.record_count} records")
    else:
        print(f"✗ Import failed: {result.errors}")


def example_error_handling():
    """Example demonstrating error handling."""
    print("\nExample: Error handling")
    print("-" * 50)
    
    # Create invalid data (missing required column)
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'product_id': ['PROD_003'] * 10
        # Missing sales_volume - will cause error
    })
    
    config = ForecastDatasetConfig(
        dataset_name='invalid_dataset',
        dataset_group_name='invalid_group'
    )
    
    forecast_integration = AmazonForecastIntegration(
        role_arn='arn:aws:iam::YOUR_ACCOUNT_ID:role/ForecastRole'
    )
    
    result = forecast_integration.import_dataset(
        data=data,
        config=config,
        product_id='PROD_003'
    )
    
    # Check for errors
    if not result.success:
        print("Expected error occurred:")
        for error in result.errors:
            print(f"  - {error}")
        print("\nThis demonstrates proper error handling.")
    else:
        print("Unexpected success - should have failed!")


if __name__ == '__main__':
    # Run main example
    # NOTE: This requires valid AWS credentials and IAM role
    # Uncomment to run:
    # main()
    
    # Run additional examples
    # example_with_minimal_features()
    # example_error_handling()
    
    print("\nTo run these examples:")
    print("1. Set up AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("2. Create an IAM role that allows Forecast to access S3")
    print("3. Update the role_arn in the examples")
    print("4. Uncomment the function calls above")
    print("5. Run: python examples/forecast_integration_usage.py")
