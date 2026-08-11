"""
Unit tests for data validation module.

Tests schema validation, data type validation, value range validation,
and duplicate detection functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.validation import DataValidator, ValidationResult, ValidationError


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create a valid dataset for testing."""
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
    
    def test_valid_data_passes_validation(self, validator, valid_data):
        """Test that valid data passes all validation checks."""
        result = validator.validate_schema(valid_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.record_count == 10
    
    def test_empty_dataframe_fails_validation(self, validator):
        """Test that empty DataFrame fails validation."""
        empty_df = pd.DataFrame()
        result = validator.validate_schema(empty_df)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "missing"
        assert "empty" in result.errors[0].message.lower()
    
    def test_none_dataframe_fails_validation(self, validator):
        """Test that None DataFrame fails validation."""
        result = validator.validate_schema(None)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "missing"
    
    def test_missing_required_column_fails_validation(self, validator, valid_data):
        """Test that missing required columns are detected."""
        # Remove timestamp column
        data_missing_timestamp = valid_data.drop(columns=['timestamp'])
        result = validator.validate_schema(data_missing_timestamp)
        
        assert result.is_valid is False
        assert any(err.field == "timestamp" and err.error_type == "missing" for err in result.errors)
    
    def test_missing_multiple_columns_fails_validation(self, validator, valid_data):
        """Test that multiple missing columns are detected."""
        # Remove multiple columns
        data_missing_cols = valid_data.drop(columns=['timestamp', 'price'])
        result = validator.validate_schema(data_missing_cols)
        
        assert result.is_valid is False
        # Should have error mentioning both missing columns
        missing_error = next(err for err in result.errors if err.error_type == "missing")
        assert "timestamp" in missing_error.field
        assert "price" in missing_error.field
    
    def test_missing_seasonality_features_fails_validation(self, validator, valid_data):
        """Test that missing all seasonality features fails validation."""
        # Remove all seasonality columns
        data_no_seasonality = valid_data.drop(columns=['day_of_week', 'month', 'quarter'])
        result = validator.validate_schema(data_no_seasonality)
        
        assert result.is_valid is False
        assert any(err.field == "seasonality_features" and err.error_type == "missing" for err in result.errors)
    
    def test_partial_seasonality_features_passes_validation(self, validator, valid_data):
        """Test that having at least one seasonality feature passes validation."""
        # Keep only month column
        data_partial_seasonality = valid_data.drop(columns=['day_of_week', 'quarter'])
        result = validator.validate_schema(data_partial_seasonality)
        
        # Should not have seasonality missing error
        assert not any(err.field == "seasonality_features" for err in result.errors)
    
    def test_invalid_timestamp_type_fails_validation(self, validator, valid_data):
        """Test that non-datetime timestamp fails validation."""
        invalid_data = valid_data.copy()
        invalid_data['timestamp'] = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', 
                                      '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
                                      '2024-01-09', '2024-01-10']  # String instead of datetime
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any(err.field == "timestamp" and err.error_type == "invalid_type" for err in result.errors)
    
    def test_empty_product_id_fails_validation(self, validator, valid_data):
        """Test that empty product_id fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'product_id'] = ''
        invalid_data.loc[1, 'product_id'] = '   '  # Whitespace only
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any(err.field == "product_id" and err.error_type == "invalid_type" for err in result.errors)
        # Should identify the specific rows
        error = next(err for err in result.errors if err.field == "product_id")
        assert 0 in error.row_indices
        assert 1 in error.row_indices
    
    def test_non_numeric_sales_volume_fails_validation(self, validator, valid_data):
        """Test that non-numeric sales_volume fails validation."""
        invalid_data = valid_data.copy()
        invalid_data['sales_volume'] = invalid_data['sales_volume'].astype(str)
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any(err.field == "sales_volume" and err.error_type == "invalid_type" for err in result.errors)
    
    def test_non_numeric_price_fails_validation(self, validator, valid_data):
        """Test that non-numeric price fails validation."""
        invalid_data = valid_data.copy()
        invalid_data['price'] = invalid_data['price'].astype(str)
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any(err.field == "price" and err.error_type == "invalid_type" for err in result.errors)
    
    def test_non_boolean_is_holiday_fails_validation(self, validator, valid_data):
        """Test that non-boolean is_holiday fails validation."""
        invalid_data = valid_data.copy()
        invalid_data['is_holiday'] = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no']
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any(err.field == "is_holiday" and err.error_type == "invalid_type" for err in result.errors)
    
    def test_negative_sales_volume_fails_validation(self, validator, valid_data):
        """Test that negative sales_volume fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'sales_volume'] = -10.0
        invalid_data.loc[5, 'sales_volume'] = -5.0
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "sales_volume" and err.error_type == "out_of_range")
        assert 0 in error.row_indices
        assert 5 in error.row_indices
    
    def test_negative_price_fails_validation(self, validator, valid_data):
        """Test that negative price fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[2, 'price'] = -1.0
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "price" and err.error_type == "out_of_range")
        assert 2 in error.row_indices
    
    def test_out_of_range_timestamp_fails_validation(self, validator, valid_data):
        """Test that timestamps outside valid range fail validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'timestamp'] = pd.Timestamp('1800-01-01')  # Too old
        invalid_data.loc[1, 'timestamp'] = pd.Timestamp('2150-01-01')  # Too far in future
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "timestamp" and err.error_type == "out_of_range")
        assert 0 in error.row_indices
        assert 1 in error.row_indices
    
    def test_invalid_day_of_week_fails_validation(self, validator, valid_data):
        """Test that day_of_week outside 0-6 range fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'day_of_week'] = -1
        invalid_data.loc[1, 'day_of_week'] = 7
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "day_of_week" and err.error_type == "out_of_range")
        assert 0 in error.row_indices
        assert 1 in error.row_indices
    
    def test_invalid_month_fails_validation(self, validator, valid_data):
        """Test that month outside 1-12 range fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'month'] = 0
        invalid_data.loc[1, 'month'] = 13
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "month" and err.error_type == "out_of_range")
        assert 0 in error.row_indices
        assert 1 in error.row_indices
    
    def test_invalid_quarter_fails_validation(self, validator, valid_data):
        """Test that quarter outside 1-4 range fails validation."""
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'quarter'] = 0
        invalid_data.loc[1, 'quarter'] = 5
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.field == "quarter" and err.error_type == "out_of_range")
        assert 0 in error.row_indices
        assert 1 in error.row_indices
    
    def test_duplicate_product_timestamp_pairs_fails_validation(self, validator, valid_data):
        """Test that duplicate (product_id, timestamp) pairs fail validation."""
        invalid_data = valid_data.copy()
        # Create duplicates by repeating first two rows
        duplicate_rows = invalid_data.iloc[[0, 1]].copy()
        invalid_data = pd.concat([invalid_data, duplicate_rows], ignore_index=True)
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        error = next(err for err in result.errors if err.error_type == "duplicate")
        assert "product_id, timestamp" in error.field
        # Should identify all 4 rows involved in duplicates (original + duplicates)
        assert len(error.row_indices) == 4
    
    def test_multiple_validation_errors_reported(self, validator, valid_data):
        """Test that multiple validation errors are all reported."""
        invalid_data = valid_data.copy()
        # Introduce multiple errors
        invalid_data.loc[0, 'sales_volume'] = -10.0  # Negative sales
        invalid_data.loc[1, 'price'] = -5.0  # Negative price
        invalid_data.loc[2, 'product_id'] = ''  # Empty product_id
        
        result = validator.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3  # At least 3 errors
        
        # Check all error types are present
        error_fields = [err.field for err in result.errors]
        assert "sales_volume" in error_fields
        assert "price" in error_fields
        assert "product_id" in error_fields
    
    def test_validation_result_record_count(self, validator, valid_data):
        """Test that ValidationResult contains correct record count."""
        result = validator.validate_schema(valid_data)
        assert result.record_count == 10
        
        # Test with different size
        small_data = valid_data.iloc[:3]
        result = validator.validate_schema(small_data)
        assert result.record_count == 3
    
    def test_zero_values_are_valid(self, validator, valid_data):
        """Test that zero values for sales_volume and price are valid."""
        data_with_zeros = valid_data.copy()
        data_with_zeros.loc[0, 'sales_volume'] = 0.0
        data_with_zeros.loc[1, 'price'] = 0.0
        
        result = validator.validate_schema(data_with_zeros)
        
        # Should not have range errors for zero values
        range_errors = [err for err in result.errors if err.error_type == "out_of_range"]
        assert not any(err.field in ["sales_volume", "price"] for err in range_errors)
