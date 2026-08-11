"""
Unit tests for training data preparation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from src.training.data_preparation import (
    TrainingDataPreparation,
    TrainingDataset,
    DataPreparationResult
)
from src.features.preprocessing import FeaturePreprocessor


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    data = {
        'timestamp': dates,
        'product_id': ['PROD_001'] * 100,
        'sales_volume': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100),
        'is_holiday': [False] * 90 + [True] * 10
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    return Mock()


@pytest.fixture
def data_preparation(mock_s3_client):
    """Create TrainingDataPreparation instance with mock S3 client."""
    return TrainingDataPreparation(s3_client=mock_s3_client)


def test_prepare_training_data_success(data_preparation, mock_s3_client, sample_historical_data):
    """Test successful training data preparation."""
    # Mock S3 response
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/product_id=PROD_001/year=2023/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    # Prepare training data
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001',
        train_split=0.8
    )
    
    # Verify success
    assert result.success is True
    assert result.dataset is not None
    assert len(result.errors) == 0
    
    # Verify dataset structure
    dataset = result.dataset
    assert isinstance(dataset, TrainingDataset)
    assert len(dataset.train_data) == 80  # 80% of 100
    assert len(dataset.validation_data) == 20  # 20% of 100
    assert dataset.target_column == 'sales_volume'
    assert len(dataset.feature_columns) > 0
    
    # Verify feature columns include engineered features
    assert 'day_of_week' in dataset.train_data.columns
    assert 'month' in dataset.train_data.columns
    assert 'quarter' in dataset.train_data.columns
    assert 'season' in dataset.train_data.columns
    
    # Verify metadata
    assert dataset.metadata['total_records'] == 100
    assert dataset.metadata['train_records'] == 80
    assert dataset.metadata['validation_records'] == 20
    assert dataset.metadata['product_id'] == 'PROD_001'


def test_prepare_training_data_invalid_split(data_preparation):
    """Test training data preparation with invalid train_split."""
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        train_split=1.5  # Invalid: > 1
    )
    
    assert result.success is False
    assert len(result.errors) > 0
    assert 'train_split must be between 0 and 1' in result.errors[0]


def test_prepare_training_data_empty_dataset(data_preparation, mock_s3_client):
    """Test training data preparation with empty dataset."""
    # Mock S3 to return no objects
    mock_s3_client.list_objects_v2.return_value = {}
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        product_id='PROD_001'
    )
    
    assert result.success is False
    assert len(result.errors) > 0
    assert 'empty' in result.errors[0].lower()


def test_prepare_training_data_missing_columns(data_preparation, mock_s3_client):
    """Test training data preparation with missing required columns."""
    # Create data missing required columns
    incomplete_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'product_id': ['PROD_001'] * 10
        # Missing sales_volume
    })
    
    parquet_buffer = BytesIO()
    incomplete_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/'
    )
    
    assert result.success is False
    assert len(result.errors) > 0
    assert 'Missing required columns' in result.errors[0]


def test_train_validation_split(data_preparation, sample_historical_data):
    """Test train/validation split functionality."""
    train_data, validation_data = data_preparation._train_validation_split(
        sample_historical_data,
        train_split=0.8
    )
    
    # Verify split sizes
    assert len(train_data) == 80
    assert len(validation_data) == 20
    
    # Verify no overlap
    assert len(set(train_data.index) & set(validation_data.index)) == 0
    
    # Verify temporal ordering (train comes before validation)
    if 'timestamp' in train_data.columns:
        assert train_data['timestamp'].max() <= validation_data['timestamp'].min()


def test_apply_feature_engineering(data_preparation, sample_historical_data):
    """Test feature engineering pipeline."""
    result = data_preparation._apply_feature_engineering(sample_historical_data)
    
    # Verify seasonality features are added
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'quarter' in result.columns
    assert 'season' in result.columns
    
    # Verify feature values are valid
    assert result['day_of_week'].min() >= 0
    assert result['day_of_week'].max() <= 6
    assert result['month'].min() >= 1
    assert result['month'].max() <= 12
    assert result['quarter'].min() >= 1
    assert result['quarter'].max() <= 4
    assert set(result['season'].unique()).issubset({'spring', 'summer', 'fall', 'winter'})


def test_identify_feature_columns(data_preparation, sample_historical_data):
    """Test feature column identification."""
    # Add engineered features
    df_engineered = data_preparation._apply_feature_engineering(sample_historical_data)
    
    feature_columns = data_preparation._identify_feature_columns(
        df_engineered,
        target_column='sales_volume'
    )
    
    # Verify feature columns
    assert 'sales_volume' not in feature_columns  # Target excluded
    assert 'timestamp' not in feature_columns  # Timestamp excluded
    assert 'product_id' not in feature_columns  # ID excluded
    assert 'season' not in feature_columns  # Categorical excluded
    
    # Verify numeric features are included
    assert 'price' in feature_columns
    assert 'is_holiday' in feature_columns
    assert 'day_of_week' in feature_columns
    assert 'month' in feature_columns
    assert 'quarter' in feature_columns


def test_load_from_s3_single_file(data_preparation, mock_s3_client, sample_historical_data):
    """Test loading single parquet file from S3."""
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation._load_from_s3(
        'historical_data/data.parquet',
        product_id=None
    )
    
    assert len(result) == 100
    assert list(result.columns) == list(sample_historical_data.columns)


def test_load_from_s3_multiple_files(data_preparation, mock_s3_client, sample_historical_data):
    """Test loading multiple parquet files from S3."""
    # Create two parquet files
    parquet_buffer1 = BytesIO()
    sample_historical_data[:50].to_parquet(parquet_buffer1, engine='pyarrow', index=False)
    parquet_buffer1.seek(0)
    
    parquet_buffer2 = BytesIO()
    sample_historical_data[50:].to_parquet(parquet_buffer2, engine='pyarrow', index=False)
    parquet_buffer2.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'historical_data/file1.parquet'},
            {'Key': 'historical_data/file2.parquet'}
        ]
    }
    
    mock_s3_client.get_object.side_effect = [
        {'Body': Mock(read=Mock(return_value=parquet_buffer1.getvalue()))},
        {'Body': Mock(read=Mock(return_value=parquet_buffer2.getvalue()))}
    ]
    
    result = data_preparation._load_from_s3(
        'historical_data/',
        product_id=None
    )
    
    assert len(result) == 100


def test_load_from_s3_with_product_filter(data_preparation, mock_s3_client, sample_historical_data):
    """Test loading data with product_id filter."""
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'historical_data/product_id=PROD_001/data.parquet'},
            {'Key': 'historical_data/product_id=PROD_002/data.parquet'}
        ]
    }
    
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation._load_from_s3(
        'historical_data/',
        product_id='PROD_001'
    )
    
    # Should only load PROD_001 file
    assert mock_s3_client.get_object.call_count == 1
    assert 'product_id=PROD_001' in mock_s3_client.get_object.call_args[1]['Key']


def test_load_dataset_for_inference(data_preparation, mock_s3_client, sample_historical_data):
    """Test loading dataset for inference (no split)."""
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/product_id=PROD_001/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.load_dataset_for_inference(
        dataset_path='historical_data/',
        product_id='PROD_001'
    )
    
    assert result.success is True
    assert result.dataset is not None
    
    # Verify no validation split
    assert len(result.dataset.train_data) == 100
    assert len(result.dataset.validation_data) == 0


def test_prepare_training_data_with_custom_target(data_preparation, mock_s3_client):
    """Test training data preparation with custom target column."""
    # Create data with custom target
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=50),
        'product_id': ['PROD_001'] * 50,
        'demand': np.random.randint(50, 200, 50),  # Custom target
        'price': np.random.uniform(10, 50, 50),
        'is_holiday': [False] * 50
    })
    
    parquet_buffer = BytesIO()
    data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        target_column='demand'
    )
    
    assert result.success is True
    assert result.dataset.target_column == 'demand'
    assert 'demand' not in result.dataset.feature_columns


def test_normalization_params_stored(data_preparation, mock_s3_client, sample_historical_data):
    """Test that normalization parameters are stored in dataset."""
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/'
    )
    
    assert result.success is True
    assert result.dataset.normalization_params is not None
    assert isinstance(result.dataset.normalization_params, dict)


def test_s3_path_parsing(data_preparation, mock_s3_client, sample_historical_data):
    """Test S3 path parsing for different formats."""
    parquet_buffer = BytesIO()
    sample_historical_data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    # Test full S3 path
    result = data_preparation._load_from_s3(
        's3://my-bucket/historical_data/data.parquet',
        product_id=None
    )
    
    # Verify correct bucket and key were used
    call_args = mock_s3_client.get_object.call_args
    assert call_args[1]['Bucket'] == 'my-bucket'
    assert call_args[1]['Key'] == 'historical_data/data.parquet'


def test_edge_case_single_record(data_preparation, mock_s3_client):
    """Test training data preparation with single record."""
    single_record = pd.DataFrame({
        'timestamp': [datetime(2023, 1, 1)],
        'product_id': ['PROD_001'],
        'sales_volume': [100],
        'price': [25.0],
        'is_holiday': [False]
    })
    
    parquet_buffer = BytesIO()
    single_record.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        train_split=0.8
    )
    
    # Should handle gracefully (train=0, validation=1 or train=1, validation=0)
    assert result.success is True
    assert result.dataset is not None


def test_edge_case_minimum_split(data_preparation, mock_s3_client):
    """Test training data preparation with very small train_split."""
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'product_id': ['PROD_001'] * 10,
        'sales_volume': np.random.randint(50, 200, 10),
        'price': np.random.uniform(10, 50, 10),
        'is_holiday': [False] * 10
    })
    
    parquet_buffer = BytesIO()
    data.to_parquet(parquet_buffer, engine='pyarrow', index=False)
    parquet_buffer.seek(0)
    
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historical_data/data.parquet'}]
    }
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=Mock(return_value=parquet_buffer.getvalue()))
    }
    
    result = data_preparation.prepare_training_data(
        dataset_path='historical_data/',
        train_split=0.1  # Very small split
    )
    
    assert result.success is True
    assert len(result.dataset.train_data) == 1
    assert len(result.dataset.validation_data) == 9
