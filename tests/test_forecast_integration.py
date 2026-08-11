"""
Unit tests for Amazon Forecast integration module.

Tests dataset import functionality including data conversion,
S3 upload, dataset group creation, and import job creation.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError

from src.training.forecast_integration import (
    AmazonForecastIntegration,
    ForecastDatasetConfig,
    ForecastImportResult
)


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'product_id': ['PROD_001'] * 100,
        'sales_volume': [100 + i * 2 for i in range(100)],
        'price': [19.99 + (i % 10) * 0.5 for i in range(100)],
        'is_holiday': [i % 7 == 0 for i in range(100)],
        'day_of_week': [i % 7 for i in range(100)],
        'month': [(i // 30) + 1 for i in range(100)],
        'quarter': [((i // 30) // 3) + 1 for i in range(100)]
    })
    
    return data


@pytest.fixture
def forecast_config():
    """Create sample Forecast configuration."""
    return ForecastDatasetConfig(
        dataset_name='test_dataset',
        dataset_group_name='test_dataset_group',
        domain='CUSTOM',
        dataset_frequency='D',
        timestamp_format='yyyy-MM-dd HH:mm:ss'
    )


@pytest.fixture
def mock_forecast_client():
    """Create mock Forecast client."""
    client = Mock()
    
    # Mock describe_dataset_group (not found)
    client.describe_dataset_group.side_effect = ClientError(
        {'Error': {'Code': 'ResourceNotFoundException'}},
        'DescribeDatasetGroup'
    )
    
    # Mock create_dataset_group
    client.create_dataset_group.return_value = {
        'DatasetGroupArn': 'arn:aws:forecast:us-east-1:123456789012:dataset-group/test_dataset_group'
    }
    
    # Mock describe_dataset (not found)
    client.describe_dataset.side_effect = ClientError(
        {'Error': {'Code': 'ResourceNotFoundException'}},
        'DescribeDataset'
    )
    
    # Mock create_dataset
    client.create_dataset.return_value = {
        'DatasetArn': 'arn:aws:forecast:us-east-1:123456789012:dataset/test_dataset'
    }
    
    # Mock create_dataset_import_job
    client.create_dataset_import_job.return_value = {
        'DatasetImportJobArn': 'arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test_dataset/test_import'
    }
    
    # Mock describe_dataset_import_job
    client.describe_dataset_import_job.return_value = {
        'Status': 'ACTIVE'
    }
    
    return client


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    client = Mock()
    client.put_object.return_value = {}
    return client


@pytest.fixture
def forecast_integration(mock_forecast_client, mock_s3_client):
    """Create AmazonForecastIntegration instance with mocked clients."""
    return AmazonForecastIntegration(
        forecast_client=mock_forecast_client,
        s3_client=mock_s3_client,
        role_arn='arn:aws:iam::123456789012:role/ForecastRole'
    )


class TestDataConversion:
    """Test data conversion to Forecast format."""
    
    def test_convert_to_forecast_format(self, forecast_integration, sample_historical_data):
        """Test conversion of historical data to Forecast CSV format."""
        result = forecast_integration._convert_to_forecast_format(
            sample_historical_data,
            'PROD_001'
        )
        
        # Verify required columns
        assert 'timestamp' in result.columns
        assert 'target_value' in result.columns
        assert 'item_id' in result.columns
        
        # Verify related features
        assert 'price' in result.columns
        assert 'is_holiday' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        
        # Verify data integrity
        assert len(result) == 100
        assert result['item_id'].unique()[0] == 'PROD_001'
        
        # Verify is_holiday is converted to integer
        assert result['is_holiday'].dtype in [int, 'int64', 'int32']
        
        # Verify sorted by timestamp
        assert result['timestamp'].is_monotonic_increasing
    
    def test_convert_filters_by_product_id(self, forecast_integration):
        """Test that conversion filters data by product_id."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='D'),
            'product_id': ['PROD_001'] * 25 + ['PROD_002'] * 25,
            'sales_volume': range(50)
        })
        
        result = forecast_integration._convert_to_forecast_format(data, 'PROD_001')
        
        assert len(result) == 25
        assert result['item_id'].unique()[0] == 'PROD_001'
    
    def test_convert_removes_duplicates(self, forecast_integration):
        """Test that conversion removes duplicate timestamps."""
        data = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'product_id': ['PROD_001'] * 3,
            'sales_volume': [100, 110, 120]
        })
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        result = forecast_integration._convert_to_forecast_format(data, 'PROD_001')
        
        # Should have only 2 records (duplicate removed)
        assert len(result) == 2
    
    def test_convert_missing_required_columns(self, forecast_integration):
        """Test error handling for missing required columns."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'product_id': ['PROD_001'] * 10
            # Missing sales_volume
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            forecast_integration._convert_to_forecast_format(data, 'PROD_001')
    
    def test_convert_with_minimal_columns(self, forecast_integration):
        """Test conversion with only required columns (no related features)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'product_id': ['PROD_001'] * 10,
            'sales_volume': range(10)
        })
        
        result = forecast_integration._convert_to_forecast_format(data, 'PROD_001')
        
        # Should have only core columns
        assert 'timestamp' in result.columns
        assert 'target_value' in result.columns
        assert 'item_id' in result.columns
        
        # Should not have related features
        assert 'price' not in result.columns
        assert 'is_holiday' not in result.columns


class TestS3Upload:
    """Test S3 upload functionality."""
    
    def test_upload_to_s3(self, forecast_integration, mock_s3_client):
        """Test uploading Forecast data to S3."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'target_value': range(10),
            'item_id': ['PROD_001'] * 10
        })
        
        s3_path = forecast_integration._upload_to_s3(
            data,
            'test_dataset',
            'PROD_001'
        )
        
        # Verify S3 path format
        assert s3_path.startswith('s3://')
        assert 'forecast_datasets/PROD_001/test_dataset' in s3_path
        assert s3_path.endswith('.csv')
        
        # Verify put_object was called
        mock_s3_client.put_object.assert_called_once()
        
        # Verify call arguments
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]['ContentType'] == 'text/csv'
        assert 'Body' in call_args[1]
    
    def test_upload_to_s3_error_handling(self, forecast_integration, mock_s3_client):
        """Test error handling for S3 upload failures."""
        mock_s3_client.put_object.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'PutObject'
        )
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'target_value': range(10),
            'item_id': ['PROD_001'] * 10
        })
        
        with pytest.raises(ClientError):
            forecast_integration._upload_to_s3(data, 'test_dataset', 'PROD_001')


class TestDatasetGroupCreation:
    """Test dataset group creation."""
    
    def test_create_dataset_group_new(self, forecast_integration, mock_forecast_client):
        """Test creating a new dataset group."""
        # Update mock to return the correct ARN for the test
        mock_forecast_client.create_dataset_group.return_value = {
            'DatasetGroupArn': 'arn:aws:forecast:us-east-1:123456789012:dataset-group/test_group'
        }
        
        arn = forecast_integration._create_dataset_group('test_group', 'CUSTOM')
        
        # Verify ARN format
        assert 'arn:aws:forecast' in arn
        assert 'dataset-group/test_group' in arn
        
        # Verify create_dataset_group was called
        mock_forecast_client.create_dataset_group.assert_called_once()
        
        call_args = mock_forecast_client.create_dataset_group.call_args
        assert call_args[1]['DatasetGroupName'] == 'test_group'
        assert call_args[1]['Domain'] == 'CUSTOM'
    
    def test_create_dataset_group_existing(self, forecast_integration, mock_forecast_client):
        """Test handling of existing dataset group."""
        # Mock describe to return existing group
        mock_forecast_client.describe_dataset_group.side_effect = None
        mock_forecast_client.describe_dataset_group.return_value = {
            'DatasetGroupArn': 'arn:aws:forecast:us-east-1:123456789012:dataset-group/existing_group'
        }
        
        arn = forecast_integration._create_dataset_group('existing_group', 'CUSTOM')
        
        # Should return existing ARN
        assert 'existing_group' in arn
        
        # Should not call create
        mock_forecast_client.create_dataset_group.assert_not_called()


class TestDatasetCreation:
    """Test dataset creation."""
    
    def test_create_dataset_new(self, forecast_integration, mock_forecast_client):
        """Test creating a new dataset with schema."""
        arn = forecast_integration._create_dataset('test_dataset', 'CUSTOM', 'D')
        
        # Verify ARN format
        assert 'arn:aws:forecast' in arn
        assert 'dataset/test_dataset' in arn
        
        # Verify create_dataset was called
        mock_forecast_client.create_dataset.assert_called_once()
        
        call_args = mock_forecast_client.create_dataset.call_args
        assert call_args[1]['DatasetName'] == 'test_dataset'
        assert call_args[1]['Domain'] == 'CUSTOM'
        assert call_args[1]['DatasetType'] == 'TARGET_TIME_SERIES'
        assert call_args[1]['DataFrequency'] == 'D'
        
        # Verify schema includes required fields
        schema = call_args[1]['Schema']
        attribute_names = [attr['AttributeName'] for attr in schema['Attributes']]
        
        assert 'timestamp' in attribute_names
        assert 'target_value' in attribute_names
        assert 'item_id' in attribute_names
        assert 'price' in attribute_names
        assert 'is_holiday' in attribute_names
        assert 'day_of_week' in attribute_names
        assert 'month' in attribute_names
        assert 'quarter' in attribute_names
    
    def test_create_dataset_existing(self, forecast_integration, mock_forecast_client):
        """Test handling of existing dataset."""
        # Mock describe to return existing dataset
        mock_forecast_client.describe_dataset.side_effect = None
        mock_forecast_client.describe_dataset.return_value = {
            'DatasetArn': 'arn:aws:forecast:us-east-1:123456789012:dataset/existing_dataset'
        }
        
        arn = forecast_integration._create_dataset('existing_dataset', 'CUSTOM', 'D')
        
        # Should return existing ARN
        assert 'existing_dataset' in arn
        
        # Should not call create
        mock_forecast_client.create_dataset.assert_not_called()


class TestImportJobCreation:
    """Test import job creation."""
    
    def test_create_import_job(self, forecast_integration, mock_forecast_client):
        """Test creating a dataset import job."""
        dataset_arn = 'arn:aws:forecast:us-east-1:123456789012:dataset/test_dataset'
        s3_path = 's3://bucket/path/data.csv'
        
        import_job_arn = forecast_integration._create_import_job(
            dataset_arn,
            s3_path,
            'test_dataset',
            'yyyy-MM-dd HH:mm:ss'
        )
        
        # Verify ARN format
        assert 'arn:aws:forecast' in import_job_arn
        assert 'dataset-import-job' in import_job_arn
        
        # Verify create_dataset_import_job was called
        mock_forecast_client.create_dataset_import_job.assert_called_once()
        
        call_args = mock_forecast_client.create_dataset_import_job.call_args
        assert call_args[1]['DatasetArn'] == dataset_arn
        assert call_args[1]['DataSource']['S3Config']['Path'] == s3_path
        assert call_args[1]['TimestampFormat'] == 'yyyy-MM-dd HH:mm:ss'
    
    def test_create_import_job_missing_role(self, mock_forecast_client, mock_s3_client):
        """Test error handling when IAM role is not provided."""
        integration = AmazonForecastIntegration(
            forecast_client=mock_forecast_client,
            s3_client=mock_s3_client,
            role_arn=None  # No role provided
        )
        
        with pytest.raises(ValueError, match="IAM role ARN is required"):
            integration._create_import_job(
                'arn:aws:forecast:us-east-1:123456789012:dataset/test',
                's3://bucket/data.csv',
                'test_dataset',
                'yyyy-MM-dd HH:mm:ss'
            )


class TestCompleteImportWorkflow:
    """Test complete dataset import workflow."""
    
    @patch('src.training.forecast_integration.boto3.client')
    def test_import_dataset_success(
        self,
        mock_boto_client,
        forecast_integration,
        sample_historical_data,
        forecast_config
    ):
        """Test successful complete dataset import workflow."""
        result = forecast_integration.import_dataset(
            sample_historical_data,
            forecast_config,
            'PROD_001'
        )
        
        # Verify success
        assert result.success is True
        assert result.dataset_group_arn is not None
        assert result.dataset_arn is not None
        assert result.import_job_arn is not None
        assert result.s3_path is not None
        assert result.record_count == 100
        assert len(result.errors) == 0
    
    def test_import_dataset_empty_data(self, forecast_integration, forecast_config):
        """Test handling of empty dataset."""
        empty_data = pd.DataFrame({
            'timestamp': [],
            'product_id': [],
            'sales_volume': []
        })
        
        result = forecast_integration.import_dataset(
            empty_data,
            forecast_config,
            'PROD_001'
        )
        
        # Should fail with appropriate error
        assert result.success is False
        assert len(result.errors) > 0
        assert result.record_count == 0
    
    def test_import_dataset_s3_error(
        self,
        forecast_integration,
        mock_s3_client,
        sample_historical_data,
        forecast_config
    ):
        """Test handling of S3 upload errors."""
        # Make S3 upload fail
        mock_s3_client.put_object.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'PutObject'
        )
        
        result = forecast_integration.import_dataset(
            sample_historical_data,
            forecast_config,
            'PROD_001'
        )
        
        # Should fail with error
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Access Denied' in result.errors[0] or 'failed' in result.errors[0].lower()


class TestImportJobPolling:
    """Test import job status polling."""
    
    def test_wait_for_import_completion_success(self, forecast_integration, mock_forecast_client):
        """Test waiting for successful import completion."""
        mock_forecast_client.describe_dataset_import_job.return_value = {
            'Status': 'ACTIVE'
        }
        
        import_job_arn = 'arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test'
        
        result = forecast_integration.wait_for_import_completion(
            import_job_arn,
            max_wait_seconds=10,
            poll_interval=1
        )
        
        assert result is True
    
    def test_wait_for_import_completion_failure(self, forecast_integration, mock_forecast_client):
        """Test waiting for failed import."""
        mock_forecast_client.describe_dataset_import_job.return_value = {
            'Status': 'CREATE_FAILED',
            'Message': 'Import failed due to invalid data'
        }
        
        import_job_arn = 'arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test'
        
        result = forecast_integration.wait_for_import_completion(
            import_job_arn,
            max_wait_seconds=10,
            poll_interval=1
        )
        
        assert result is False
    
    def test_wait_for_import_completion_timeout(self, forecast_integration, mock_forecast_client):
        """Test timeout while waiting for import."""
        # Always return IN_PROGRESS status
        mock_forecast_client.describe_dataset_import_job.return_value = {
            'Status': 'CREATE_IN_PROGRESS'
        }
        
        import_job_arn = 'arn:aws:forecast:us-east-1:123456789012:dataset-import-job/test'
        
        result = forecast_integration.wait_for_import_completion(
            import_job_arn,
            max_wait_seconds=2,
            poll_interval=1
        )
        
        assert result is False


class TestARNBuilding:
    """Test ARN construction."""
    
    @patch('src.training.forecast_integration.boto3.client')
    def test_build_arn(self, mock_boto_client, forecast_integration):
        """Test building Forecast ARN."""
        # Mock STS client
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {'Account': '123456789012'}
        mock_boto_client.return_value = mock_sts
        
        arn = forecast_integration._build_arn('dataset', 'test_dataset')
        
        assert arn.startswith('arn:aws:forecast:')
        assert ':dataset/test_dataset' in arn
