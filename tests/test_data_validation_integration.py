"""
Integration tests for data validation module.

Tests realistic scenarios with various data quality issues.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.data import DataValidator, ValidationResult, ValidationError


class TestDataValidationIntegration:
    """Integration tests for data validation scenarios."""
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator()
    
    def test_realistic_valid_dataset(self, validator):
        """Test validation with a realistic valid dataset."""
        # Create a realistic dataset with multiple products
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D').tolist() * 3,
            'product_id': ['PROD_A'] * 30 + ['PROD_B'] * 30 + ['PROD_C'] * 30,
            'sales_volume': [100 + i * 2 for i in range(90)],
            'price': [10.0 + (i % 5) * 0.5 for i in range(90)],
            'is_holiday': [i % 7 == 0 for i in range(90)],
            'day_of_week': [(i % 7) for i in range(90)],
            'month': [1] * 30 + [2] * 30 + [3] * 30,
            'quarter': [1] * 90
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.record_count == 90
    
    def test_dataset_with_season_column_instead_of_numeric_features(self, validator):
        """Test that season column (categorical) is accepted as seasonality feature."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'product_id': ['PROD_001'] * 10,
            'sales_volume': [100.0] * 10,
            'price': [10.0] * 10,
            'is_holiday': [False] * 10,
            'season': ['winter'] * 10  # Only season column, no day_of_week/month/quarter
        })
        
        result = validator.validate_schema(data)
        
        # Should pass because season is a valid seasonality feature
        assert result.is_valid is True
        assert not any(err.field == "seasonality_features" for err in result.errors)
    
    def test_dataset_with_multiple_data_quality_issues(self, validator):
        """Test dataset with multiple realistic data quality issues."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
            'product_id': ['PROD_A', '', 'PROD_C', 'PROD_D', 'PROD_E'],  # Empty product_id
            'sales_volume': [100.0, 150.0, -50.0, 200.0, 175.0],  # Negative sales
            'price': [10.0, 11.0, 12.0, -5.0, 10.5],  # Negative price
            'is_holiday': [False, True, False, False, True],
            'month': [1, 2, 3, 4, 5]
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 3  # Empty product_id, negative sales, negative price
        
        # Verify specific errors
        error_fields = [err.field for err in result.errors]
        assert 'product_id' in error_fields
        assert 'sales_volume' in error_fields
        assert 'price' in error_fields
    
    def test_dataset_with_duplicate_records(self, validator):
        """Test dataset with duplicate (product_id, timestamp) pairs."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03']),
            'product_id': ['PROD_A', 'PROD_A', 'PROD_A', 'PROD_A'],  # First two are duplicates
            'sales_volume': [100.0, 150.0, 200.0, 175.0],
            'price': [10.0, 10.5, 11.0, 10.5],
            'is_holiday': [False, False, True, False],
            'month': [1, 1, 1, 1]
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is False
        duplicate_error = next(err for err in result.errors if err.error_type == "duplicate")
        assert len(duplicate_error.row_indices) == 2  # Both duplicate rows identified
    
    def test_large_dataset_validation_performance(self, validator):
        """Test validation performance with a large dataset."""
        # Create a large dataset (100k records)
        num_records = 100000
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=num_records, freq='H'),
            'product_id': [f'PROD_{i % 100}' for i in range(num_records)],
            'sales_volume': [100.0 + i % 50 for i in range(num_records)],
            'price': [10.0 + (i % 10) * 0.5 for i in range(num_records)],
            'is_holiday': [i % 100 == 0 for i in range(num_records)],
            'day_of_week': [i % 7 for i in range(num_records)],
            'month': [(i % 12) + 1 for i in range(num_records)],
            'quarter': [((i % 12) // 3) + 1 for i in range(num_records)]
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is True
        assert result.record_count == num_records
    
    def test_edge_case_boundary_values(self, validator):
        """Test validation with boundary values for ranges."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['1900-01-01', '2100-12-31', '2024-06-15']),
            'product_id': ['PROD_A', 'PROD_B', 'PROD_C'],
            'sales_volume': [0.0, 1000000.0, 500.0],  # Zero and very large values
            'price': [0.0, 10000.0, 50.0],  # Zero and very large values
            'is_holiday': [False, True, False],
            'day_of_week': [0, 6, 3],  # Boundary values 0 and 6
            'month': [1, 12, 6],  # Boundary values 1 and 12
            'quarter': [1, 4, 2]  # Boundary values 1 and 4
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_mixed_product_ids_with_different_time_ranges(self, validator):
        """Test validation with multiple products having different time ranges."""
        # Product A: Jan 2024
        product_a = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'product_id': ['PROD_A'] * 10,
            'sales_volume': [100.0] * 10,
            'price': [10.0] * 10,
            'is_holiday': [False] * 10,
            'month': [1] * 10
        })
        
        # Product B: Feb 2024
        product_b = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-02-01', periods=10, freq='D'),
            'product_id': ['PROD_B'] * 10,
            'sales_volume': [200.0] * 10,
            'price': [20.0] * 10,
            'is_holiday': [False] * 10,
            'month': [2] * 10
        })
        
        data = pd.concat([product_a, product_b], ignore_index=True)
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is True
        assert result.record_count == 20
    
    def test_validation_error_messages_are_descriptive(self, validator):
        """Test that validation error messages are clear and actionable."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01']),
            'product_id': [''],  # Empty
            'sales_volume': [-10.0],  # Negative
            'price': [10.0],
            'is_holiday': [False],
            'month': [1]
        })
        
        result = validator.validate_schema(data)
        
        assert result.is_valid is False
        
        # Check that error messages are descriptive
        for error in result.errors:
            assert error.field is not None and error.field != ""
            assert error.error_type in ["missing", "invalid_type", "out_of_range", "duplicate"]
            assert error.message is not None and len(error.message) > 0
            assert "must" in error.message.lower() or "required" in error.message.lower()
