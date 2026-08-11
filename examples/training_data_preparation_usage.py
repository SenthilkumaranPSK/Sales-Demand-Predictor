"""
Example usage of the Training Data Preparation module.

This example demonstrates how to prepare training data for model training
by loading historical datasets from S3, applying feature engineering,
and splitting into train/validation sets.
"""

from src.training.data_preparation import TrainingDataPreparation
import pandas as pd


def example_prepare_training_data():
    """Example: Prepare training data from S3 dataset."""
    print("=" * 60)
    print("Example: Prepare Training Data from S3")
    print("=" * 60)
    
    # Initialize the data preparation service
    data_prep = TrainingDataPreparation()
    
    # Prepare training data from S3
    # This will:
    # 1. Load historical data from S3
    # 2. Extract seasonality features (day_of_week, month, quarter, season)
    # 3. Preprocess features (validate holidays, prices, normalize)
    # 4. Split into train (80%) and validation (20%) sets
    result = data_prep.prepare_training_data(
        dataset_path='s3://my-bucket/historical_data/',
        product_id='PROD_001',
        train_split=0.8,
        target_column='sales_volume'
    )
    
    if result.success:
        dataset = result.dataset
        
        print(f"\n✓ Training data prepared successfully!")
        print(f"  Total records: {dataset.metadata['total_records']}")
        print(f"  Train records: {dataset.metadata['train_records']}")
        print(f"  Validation records: {dataset.metadata['validation_records']}")
        print(f"  Target column: {dataset.target_column}")
        print(f"  Feature columns: {dataset.feature_columns}")
        
        # Access train and validation data
        print(f"\nTrain data shape: {dataset.train_data.shape}")
        print(f"Validation data shape: {dataset.validation_data.shape}")
        
        # Access normalization parameters (for later use in inference)
        print(f"\nNormalization parameters: {dataset.normalization_params}")
        
        # Example: Get features and target for training
        X_train = dataset.train_data[dataset.feature_columns]
        y_train = dataset.train_data[dataset.target_column]
        
        X_val = dataset.validation_data[dataset.feature_columns]
        y_val = dataset.validation_data[dataset.target_column]
        
        print(f"\nX_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        
    else:
        print(f"\n✗ Training data preparation failed!")
        for error in result.errors:
            print(f"  Error: {error}")


def example_custom_split_ratio():
    """Example: Prepare training data with custom split ratio."""
    print("\n" + "=" * 60)
    print("Example: Custom Train/Validation Split")
    print("=" * 60)
    
    data_prep = TrainingDataPreparation()
    
    # Use 90/10 split instead of default 80/20
    result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_002',
        train_split=0.9,  # 90% for training
        target_column='sales_volume'
    )
    
    if result.success:
        dataset = result.dataset
        print(f"\n✓ Custom split applied successfully!")
        print(f"  Train: {len(dataset.train_data)} records (90%)")
        print(f"  Validation: {len(dataset.validation_data)} records (10%)")
    else:
        print(f"\n✗ Failed: {result.errors}")


def example_custom_target_column():
    """Example: Prepare training data with custom target column."""
    print("\n" + "=" * 60)
    print("Example: Custom Target Column")
    print("=" * 60)
    
    data_prep = TrainingDataPreparation()
    
    # Use 'demand' as target instead of 'sales_volume'
    result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_003',
        train_split=0.8,
        target_column='demand'  # Custom target column
    )
    
    if result.success:
        dataset = result.dataset
        print(f"\n✓ Custom target column used!")
        print(f"  Target column: {dataset.target_column}")
        print(f"  Feature columns: {dataset.feature_columns}")
    else:
        print(f"\n✗ Failed: {result.errors}")


def example_load_for_inference():
    """Example: Load dataset for inference (no split)."""
    print("\n" + "=" * 60)
    print("Example: Load Dataset for Inference")
    print("=" * 60)
    
    data_prep = TrainingDataPreparation()
    
    # Load dataset without train/validation split
    # Useful for inference or when you want to use all data
    result = data_prep.load_dataset_for_inference(
        dataset_path='historical_data/',
        product_id='PROD_001'
    )
    
    if result.success:
        dataset = result.dataset
        print(f"\n✓ Dataset loaded for inference!")
        print(f"  Total records: {len(dataset.train_data)}")
        print(f"  Feature columns: {dataset.feature_columns}")
        print(f"  No validation split (validation_data is empty)")
    else:
        print(f"\n✗ Failed: {result.errors}")


def example_feature_engineering_details():
    """Example: Show feature engineering details."""
    print("\n" + "=" * 60)
    print("Example: Feature Engineering Details")
    print("=" * 60)
    
    data_prep = TrainingDataPreparation()
    
    result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    if result.success:
        dataset = result.dataset
        
        print("\n✓ Feature Engineering Applied:")
        print("\n1. Seasonality Features Extracted:")
        print("   - day_of_week: 0-6 (Monday=0, Sunday=6)")
        print("   - month: 1-12")
        print("   - quarter: 1-4")
        print("   - season: spring, summer, fall, winter")
        
        print("\n2. Feature Preprocessing:")
        print("   - Holiday indicators validated (boolean)")
        print("   - Price data validated (numeric, non-negative)")
        print("   - Features normalized (standardization)")
        
        print("\n3. Available Features:")
        for feature in dataset.feature_columns:
            print(f"   - {feature}")
        
        print("\n4. Normalization Parameters (for inference):")
        for feature, params in dataset.normalization_params.items():
            print(f"   - {feature}: mean={params['mean']:.2f}, std={params['std']:.2f}")
    else:
        print(f"\n✗ Failed: {result.errors}")


def example_error_handling():
    """Example: Error handling scenarios."""
    print("\n" + "=" * 60)
    print("Example: Error Handling")
    print("=" * 60)
    
    data_prep = TrainingDataPreparation()
    
    # Example 1: Invalid train_split
    print("\n1. Invalid train_split (> 1.0):")
    result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        train_split=1.5  # Invalid
    )
    if not result.success:
        print(f"   ✓ Error caught: {result.errors[0]}")
    
    # Example 2: Invalid train_split (< 0)
    print("\n2. Invalid train_split (< 0):")
    result = data_prep.prepare_training_data(
        dataset_path='historical_data/',
        train_split=-0.2  # Invalid
    )
    if not result.success:
        print(f"   ✓ Error caught: {result.errors[0]}")
    
    # Example 3: Empty dataset
    print("\n3. Empty dataset:")
    result = data_prep.prepare_training_data(
        dataset_path='nonexistent_path/',
        product_id='NONEXISTENT'
    )
    if not result.success:
        print(f"   ✓ Error caught: {result.errors[0]}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Training Data Preparation Usage Examples")
    print("=" * 60)
    
    # Note: These examples assume you have:
    # 1. AWS credentials configured
    # 2. Historical data stored in S3
    # 3. Data in the correct format with required columns
    
    print("\nNote: These examples require actual S3 data to run.")
    print("They demonstrate the API usage patterns.\n")
    
    # Uncomment to run examples (requires actual S3 data):
    # example_prepare_training_data()
    # example_custom_split_ratio()
    # example_custom_target_column()
    # example_load_for_inference()
    # example_feature_engineering_details()
    # example_error_handling()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
