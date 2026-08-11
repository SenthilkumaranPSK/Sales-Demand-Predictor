"""
Unit tests for data ingestion service.

Tests CSV and JSON parsing, format detection, validation integration,
S3 storage, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import json
from unittest.mock import Mock, MagicMock, patch, call

from src.data.ingestion import DataIngestionService, IngestionResult
from src.data.validation import DataValidator, ValidationResult, ValidationError


class TestDataIngestionService:
    """Test suite for DataIngestionService class."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client for testing."""
        mock_client = Mock()
        mock_client.put_object = Mock(return_value={'ETag': 'test-etag'})
        return mock_client
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator()
    
    @pytest.fixture
    def ingestion_service(self, mock_s3_client, validator):
        """Create a DataIngestionService instance with mocked S3."""
        return DataIngestionService(s3_client=mock_s3_client, validator=validator)
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame for testing."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'timestamp': dates,
            'product_id': ['PROD_001'] * 10,
            'sales_volume': [100.0, 150.0, 200.0, 175.0, 225.0, 190.0, 210.0, 180.0, 195.0, 205.0],
            'price': [10.0, 10.5, 10.0, 10.5, 11.0, 10.5, 10.0, 10.5, 11.0, 10.5],
            'is_holiday': [False, False, True, False, False, False, True, False, False, False],
            'day_of_week': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            'month': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'quarter': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
    
    @pytest.fixture
    def valid_csv_string(self, valid_dataframe):
        """Create a valid CSV string for testing."""
        return valid_dataframe.to_csv(index=False)
    
    @pytest.fixture
    def valid_json_string(self, valid_dataframe):
        """Create a valid JSON string for testing."""
        # Convert DataFrame to list of dicts
        records = valid_dataframe.to_dict(orient='records')
        # Convert timestamps to ISO format strings
        for record in records:
            record['timestamp'] = record['timestamp'].isoformat()
        return json.dumps(records)
    
    def test_ingest_valid_dataframe(self, ingestion_service, valid_dataframe):
        """Test ingesting a valid DataFrame."""
        result = ingestion_service.ingest_batch(valid_dataframe, format="auto")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
        assert result.validation_result is not None
        assert result.validation_result.is_valid is True
        assert len(result.errors) == 0
        assert result.ingestion_time_seconds > 0
    
    def test_ingest_valid_csv_string(self, ingestion_service, valid_csv_string):
        """Test ingesting a valid CSV string."""
        result = ingestion_service.ingest_batch(valid_csv_string, format="csv")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
        assert len(result.errors) == 0
    
    def test_ingest_valid_csv_bytes(self, ingestion_service, valid_csv_string):
        """Test ingesting valid CSV bytes."""
        csv_bytes = valid_csv_string.encode('utf-8')
        result = ingestion_service.ingest_batch(csv_bytes, format="csv")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
    
    def test_ingest_valid_json_string(self, ingestion_service, valid_json_string):
        """Test ingesting a valid JSON string."""
        result = ingestion_service.ingest_batch(valid_json_string, format="json")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
        assert len(result.errors) == 0
    
    def test_ingest_valid_json_bytes(self, ingestion_service, valid_json_string):
        """Test ingesting valid JSON bytes."""
        json_bytes = valid_json_string.encode('utf-8')
        result = ingestion_service.ingest_batch(json_bytes, format="json")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
    
    def test_ingest_list_of_dicts(self, ingestion_service, valid_dataframe):
        """Test ingesting a list of dictionaries."""
        records = valid_dataframe.to_dict(orient='records')
        result = ingestion_service.ingest_batch(records, format="auto")
        
        assert result.success is True
        assert result.record_count == 10
        assert result.s3_path is not None
    
    def test_format_auto_detection_csv(self, ingestion_service, valid_csv_string):
        """Test automatic format detection for CSV."""
        result = ingestion_service.ingest_batch(valid_csv_string, format="auto")
        
        assert result.success is True
        assert result.record_count == 10
    
    def test_format_auto_detection_json(self, ingestion_service, valid_json_string):
        """Test automatic format detection for JSON."""
        result = ingestion_service.ingest_batch(valid_json_string, format="auto")
        
        assert result.success is True
        assert result.record_count == 10
    
    def test_ingest_invalid_data_fails_validation(self, ingestion_service):
        """Test that invalid data fails validation."""
        # Create DataFrame with missing required column
        invalid_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5),
            'product_id': ['PROD_001'] * 5,
            # Missing sales_volume, price, is_holiday, seasonality features
        })
        
        result = ingestion_service.ingest_batch(invalid_df, format="auto")
        
        assert result.success is False
        assert result.record_count == 5
        assert len(result.errors) > 0
        assert result.validation_result is not None
        assert result.validation_result.is_valid is False
    
    def test_ingest_data_with_negative_values_fails(self, ingestion_service, valid_dataframe):
        """Test that data with negative values fails validation."""
        invalid_df = valid_dataframe.copy()
        invalid_df.loc[0, 'sales_volume'] = -10.0
        
        result = ingestion_service.ingest_batch(invalid_df, format="auto")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert any('sales_volume' in err for err in result.errors)
    
    def test_ingest_empty_dataframe(self, ingestion_service):
        """Test ingesting an empty DataFrame."""
        empty_df = pd.DataFrame()
        result = ingestion_service.ingest_batch(empty_df, format="auto")
        
        assert result.success is False
        assert result.record_count == 0
        assert len(result.errors) > 0
    
    def test_ingest_exceeds_max_batch_size(self, ingestion_service, valid_dataframe):
        """Test that exceeding max batch size is rejected."""
        # Create a large DataFrame by repeating valid data
        large_df = pd.concat([valid_dataframe] * 600000, ignore_index=True)  # > 5M records
        
        result = ingestion_service.ingest_batch(large_df, format="auto")
        
        assert result.success is False
        assert "exceeds maximum" in result.errors[0]
    
    def test_parse_malformed_csv_fails(self, ingestion_service):
        """Test that malformed CSV fails gracefully."""
        malformed_csv = "timestamp,product_id,sales_volume\n2024-01-01,PROD_001,100\n2024-01-02,PROD_002"  # Missing value
        
        result = ingestion_service.ingest_batch(malformed_csv, format="csv")
        
        # Should either fail parsing or fail validation
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_parse_malformed_json_fails(self, ingestion_service):
        """Test that malformed JSON fails gracefully."""
        malformed_json = '{"timestamp": "2024-01-01", "product_id": "PROD_001"'  # Missing closing brace
        
        result = ingestion_service.ingest_batch(malformed_json, format="json")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert any('JSON' in err or 'parse' in err for err in result.errors)
    
    def test_parse_json_with_records_key(self, ingestion_service, valid_dataframe):
        """Test parsing JSON with 'records' key."""
        records = valid_dataframe.to_dict(orient='records')
        for record in records:
            record['timestamp'] = record['timestamp'].isoformat()
        json_with_records = json.dumps({'records': records})
        
        result = ingestion_service.ingest_batch(json_with_records, format="json")
        
        assert result.success is True
        assert result.record_count == 10
    
    def test_parse_json_single_record(self, ingestion_service):
        """Test parsing JSON with a single record (dict)."""
        single_record = {
            'timestamp': '2024-01-01T00:00:00',
            'product_id': 'PROD_001',
            'sales_volume': 100.0,
            'price': 10.0,
            'is_holiday': False,
            'day_of_week': 0,
            'month': 1,
            'quarter': 1
        }
        json_string = json.dumps(single_record)
        
        result = ingestion_service.ingest_batch(json_string, format="json")
        
        assert result.success is True
        assert result.record_count == 1
    
    def test_normalize_timestamp_conversion(self, ingestion_service):
        """Test that timestamp strings are converted to datetime."""
        data = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'product_id': ['PROD_001'] * 3,
            'sales_volume': [100.0, 150.0, 200.0],
            'price': [10.0, 10.5, 11.0],
            'is_holiday': [False, False, True],
            'month': [1, 1, 1]
        })
        
        result = ingestion_service.ingest_batch(data, format="auto")
        
        # Should succeed after normalization
        assert result.success is True
    
    def test_normalize_boolean_conversion(self, ingestion_service):
        """Test that various boolean representations are converted."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=6),
            'product_id': ['PROD_001'] * 6,
            'sales_volume': [100.0] * 6,
            'price': [10.0] * 6,
            'is_holiday': [1, 0, 'true', 'false', 'yes', 'no'],  # Various boolean formats
            'month': [1] * 6
        })
        
        result = ingestion_service.ingest_batch(data, format="auto")
        
        # Should succeed after normalization
        assert result.success is True
    
    def test_normalize_numeric_conversion(self, ingestion_service):
        """Test that numeric strings are converted to numbers."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=3),
            'product_id': ['PROD_001'] * 3,
            'sales_volume': ['100', '150', '200'],  # String numbers
            'price': ['10.0', '10.5', '11.0'],  # String numbers
            'is_holiday': [False, False, True],
            'month': ['1', '1', '1']  # String numbers
        })
        
        result = ingestion_service.ingest_batch(data, format="auto")
        
        # Should succeed after normalization
        assert result.success is True
    
    def test_s3_storage_partitioning(self, ingestion_service, valid_dataframe, mock_s3_client):
        """Test that data is partitioned by product_id and year in S3."""
        result = ingestion_service.ingest_batch(valid_dataframe, format="auto")
        
        assert result.success is True
        
        # Verify S3 put_object was called
        assert mock_s3_client.put_object.called
        
        # Check that the S3 key includes partitioning
        call_args = mock_s3_client.put_object.call_args
        s3_key = call_args[1]['Key']
        assert 'product_id=' in s3_key
        assert 'year=' in s3_key
        assert s3_key.endswith('.parquet')
    
    def test_s3_storage_multiple_products(self, ingestion_service, mock_s3_client):
        """Test that multiple products are stored in separate partitions."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=6),
            'product_id': ['PROD_001', 'PROD_001', 'PROD_001', 'PROD_002', 'PROD_002', 'PROD_002'],
            'sales_volume': [100.0] * 6,
            'price': [10.0] * 6,
            'is_holiday': [False] * 6,
            'month': [1] * 6
        })
        
        result = ingestion_service.ingest_batch(data, format="auto")
        
        assert result.success is True
        
        # Should have called put_object twice (once per product)
        assert mock_s3_client.put_object.call_count == 2
        
        # Check that both products are in the keys
        call_args_list = mock_s3_client.put_object.call_args_list
        keys = [call[1]['Key'] for call in call_args_list]
        assert any('product_id=PROD_001' in key for key in keys)
        assert any('product_id=PROD_002' in key for key in keys)
    
    def test_s3_storage_multiple_years(self, ingestion_service, mock_s3_client):
        """Test that data spanning multiple years is partitioned by year."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-12-31', '2024-01-01', '2024-01-02']),
            'product_id': ['PROD_001'] * 3,
            'sales_volume': [100.0] * 3,
            'price': [10.0] * 3,
            'is_holiday': [False] * 3,
            'month': [12, 1, 1]
        })
        
        result = ingestion_service.ingest_batch(data, format="auto")
        
        assert result.success is True
        
        # Should have called put_object twice (once per year)
        assert mock_s3_client.put_object.call_count == 2
        
        # Check that both years are in the keys
        call_args_list = mock_s3_client.put_object.call_args_list
        keys = [call[1]['Key'] for call in call_args_list]
        assert any('year=2023' in key for key in keys)
        assert any('year=2024' in key for key in keys)
    
    def test_s3_storage_failure_returns_error(self, ingestion_service, valid_dataframe, mock_s3_client):
        """Test that S3 storage failures are handled gracefully."""
        # Make S3 put_object raise an exception
        mock_s3_client.put_object.side_effect = Exception("S3 connection failed")
        
        result = ingestion_service.ingest_batch(valid_dataframe, format="auto")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert any('S3' in err or 'failed' in err for err in result.errors)
    
    def test_unsupported_data_type_fails(self, ingestion_service):
        """Test that unsupported data types fail gracefully."""
        unsupported_data = 12345  # Integer instead of DataFrame, list, str, or bytes
        
        result = ingestion_service.ingest_batch(unsupported_data, format="auto")
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_unsupported_format_fails(self, ingestion_service, valid_csv_string):
        """Test that unsupported format specification fails."""
        result = ingestion_service.ingest_batch(valid_csv_string, format="xml")
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_ingestion_time_recorded(self, ingestion_service, valid_dataframe):
        """Test that ingestion time is recorded."""
        result = ingestion_service.ingest_batch(valid_dataframe, format="auto")
        
        assert result.success is True
        assert result.ingestion_time_seconds > 0
        assert result.ingestion_time_seconds < 10  # Should be fast for small dataset
    
    def test_validation_errors_included_in_result(self, ingestion_service):
        """Test that validation errors are included in the result."""
        invalid_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=3),
            'product_id': ['PROD_001'] * 3,
            'sales_volume': [-10.0, 150.0, 200.0],  # Negative value
            'price': [10.0, -5.0, 11.0],  # Negative value
            'is_holiday': [False, False, True],
            'month': [1, 1, 1]
        })
        
        result = ingestion_service.ingest_batch(invalid_df, format="auto")
        
        assert result.success is False
        assert result.validation_result is not None
        assert len(result.validation_result.errors) > 0
        assert len(result.errors) > 0
        # Check that error messages mention the fields
        error_text = ' '.join(result.errors)
        assert 'sales_volume' in error_text or 'price' in error_text
    
    def test_empty_list_returns_empty_dataframe(self, ingestion_service):
        """Test that empty list is handled correctly."""
        result = ingestion_service.ingest_batch([], format="auto")
        
        assert result.success is False
        assert result.record_count == 0
    
    def test_list_with_non_dict_items_fails(self, ingestion_service):
        """Test that list with non-dict items fails."""
        invalid_list = [1, 2, 3, 4, 5]
        
        result = ingestion_service.ingest_batch(invalid_list, format="auto")
        
        assert result.success is False
        assert len(result.errors) > 0
