"""
Unit tests for feature preprocessing.

Tests the FeaturePreprocessor class to ensure correct validation and normalization
of holiday indicators and price data.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.preprocessing import FeaturePreprocessor, PreprocessingResult


class TestFeaturePreprocessor:
    """Test suite for feature preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a feature preprocessor instance."""
        return FeaturePreprocessor()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sample data."""
        return pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17']),
            'product_id': ['PROD_001', 'PROD_001', 'PROD_001'],
            'sales_volume': [100.0, 150.0, 200.0],
            'price': [10.0, 12.0, 11.0],
            'is_holiday': [False, True, False]
        })
    
    def test_preprocess_valid_data(self, preprocessor, valid_data):
        """Test preprocessing with valid data."""
        result = preprocessor.preprocess_features(valid_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.data is not None
        assert len(result.data) == len(valid_data)
    
    def test_holiday_validation_boolean_type(self, preprocessor):
        """Test that boolean holiday indicators are accepted."""
        data = pd.DataFrame({
            'is_holiday': [True, False, True, False],
            'price': [10.0, 12.0, 11.0, 13.0]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
    
    def test_holiday_validation_numeric_conversion(self, preprocessor):
        """Test that numeric holiday indicators (0, 1) are converted to boolean."""
        data = pd.DataFrame({
            'is_holiday': [0, 1, 0, 1],
            'price': [10.0, 12.0, 11.0, 13.0]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
        assert result.data['is_holiday'].tolist() == [False, True, False, True]
    
    def test_holiday_validation_string_conversion(self, preprocessor):
        """Test that string holiday indicators are converted to boolean."""
        data = pd.DataFrame({
            'is_holiday': ['true', 'false', 'True', 'False'],
            'price': [10.0, 12.0, 11.0, 13.0]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
        assert result.data['is_holiday'].tolist() == [True, False, True, False]
    
    def test_holiday_validation_invalid_values(self, preprocessor):
        """Test that invalid holiday values are rejected."""
        data = pd.DataFrame({
            'is_holiday': ['yes', 'no', 'maybe', 'true'],
            'price': [10.0, 12.0, 11.0, 13.0]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('invalid values' in error.lower() for error in result.errors)
    
    def test_holiday_validation_missing_values(self, preprocessor):
        """Test that missing holiday values are rejected."""
        data = pd.DataFrame({
            'is_holiday': [True, False, None, True],
            'price': [10.0, 12.0, 11.0, 13.0]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('missing values' in error.lower() for error in result.errors)
    
    def test_price_validation_numeric_type(self, preprocessor):
        """Test that numeric price data is accepted."""
        data = pd.DataFrame({
            'price': [10.0, 12.5, 11.0, 13.99],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_price_validation_non_numeric_type(self, preprocessor):
        """Test that non-numeric price data is rejected."""
        data = pd.DataFrame({
            'price': ['10.0', '12.5', '11.0', '13.99'],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('numeric type' in error.lower() for error in result.errors)
    
    def test_price_validation_negative_values(self, preprocessor):
        """Test that negative price values are rejected."""
        data = pd.DataFrame({
            'price': [10.0, -5.0, 11.0, 13.0],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('negative' in error.lower() for error in result.errors)
    
    def test_price_validation_zero_allowed(self, preprocessor):
        """Test that zero price is allowed (non-negative includes zero)."""
        data = pd.DataFrame({
            'price': [10.0, 0.0, 11.0, 13.0],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_price_validation_missing_values(self, preprocessor):
        """Test that missing price values are rejected."""
        data = pd.DataFrame({
            'price': [10.0, np.nan, 11.0, 13.0],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('missing values' in error.lower() for error in result.errors)
    
    def test_price_validation_infinite_values(self, preprocessor):
        """Test that infinite price values are rejected."""
        data = pd.DataFrame({
            'price': [10.0, np.inf, 11.0, 13.0],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('infinite' in error.lower() for error in result.errors)
    
    def test_normalization_enabled(self, preprocessor, valid_data):
        """Test that normalization is applied when enabled."""
        result = preprocessor.preprocess_features(valid_data, normalize=True)
        
        assert result.is_valid
        assert 'price_normalized' in result.data.columns
        assert 'sales_volume_normalized' in result.data.columns
        assert len(result.normalization_params) > 0
    
    def test_normalization_disabled(self, preprocessor, valid_data):
        """Test that normalization is skipped when disabled."""
        result = preprocessor.preprocess_features(valid_data, normalize=False)
        
        assert result.is_valid
        assert 'price_normalized' not in result.data.columns
        assert 'sales_volume_normalized' not in result.data.columns
    
    def test_normalization_parameters_stored(self, preprocessor, valid_data):
        """Test that normalization parameters are stored."""
        result = preprocessor.preprocess_features(valid_data, normalize=True, fit_normalization=True)
        
        assert 'price' in result.normalization_params
        assert 'mean' in result.normalization_params['price']
        assert 'std' in result.normalization_params['price']
        
        # Verify parameters are correct
        expected_mean = valid_data['price'].mean()
        expected_std = valid_data['price'].std()
        
        assert abs(result.normalization_params['price']['mean'] - expected_mean) < 1e-6
        assert abs(result.normalization_params['price']['std'] - expected_std) < 1e-6
    
    def test_normalization_standardization(self, preprocessor, valid_data):
        """Test that normalization uses standardization (z-score)."""
        result = preprocessor.preprocess_features(valid_data, normalize=True)
        
        # Check that normalized values have mean ~0 and std ~1
        normalized_price = result.data['price_normalized']
        
        assert abs(normalized_price.mean()) < 1e-6
        assert abs(normalized_price.std() - 1.0) < 1e-6
    
    def test_normalization_fit_vs_transform(self, preprocessor):
        """Test fitting normalization parameters vs. transforming with existing parameters."""
        train_data = pd.DataFrame({
            'price': [10.0, 12.0, 11.0, 13.0],
            'is_holiday': [True, False, True, False]
        })
        
        test_data = pd.DataFrame({
            'price': [10.5, 11.5],
            'is_holiday': [False, True]
        })
        
        # Fit on training data
        train_result = preprocessor.preprocess_features(train_data, normalize=True, fit_normalization=True)
        
        # Transform test data using fitted parameters
        test_result = preprocessor.preprocess_features(test_data, normalize=True, fit_normalization=False)
        
        # Verify same normalization parameters are used
        assert train_result.normalization_params == test_result.normalization_params
        
        # Verify test data is normalized using training parameters
        train_mean = train_result.normalization_params['price']['mean']
        train_std = train_result.normalization_params['price']['std']
        
        expected_normalized = (test_data['price'] - train_mean) / train_std
        actual_normalized = test_result.data['price_normalized']
        
        assert np.allclose(expected_normalized, actual_normalized)
    
    def test_normalization_constant_feature(self, preprocessor):
        """Test normalization handles constant features (zero std)."""
        data = pd.DataFrame({
            'price': [10.0, 10.0, 10.0, 10.0],
            'is_holiday': [True, False, True, False]
        })
        
        result = preprocessor.preprocess_features(data, normalize=True)
        
        assert result.is_valid
        assert 'price_normalized' in result.data.columns
        # With zero std, normalization should use std=1.0
        assert result.normalization_params['price']['std'] == 1.0
    
    def test_denormalize_feature(self, preprocessor, valid_data):
        """Test denormalization of feature values."""
        # Fit normalization
        result = preprocessor.preprocess_features(valid_data, normalize=True)
        
        # Get normalized values
        normalized_values = result.data['price_normalized'].values
        
        # Denormalize
        denormalized_values = preprocessor.denormalize_feature(normalized_values, 'price')
        
        # Verify denormalized values match original
        original_values = valid_data['price'].values
        assert np.allclose(denormalized_values, original_values)
    
    def test_get_normalization_params(self, preprocessor, valid_data):
        """Test getting normalization parameters."""
        preprocessor.preprocess_features(valid_data, normalize=True)
        
        params = preprocessor.get_normalization_params()
        
        assert 'price' in params
        assert 'mean' in params['price']
        assert 'std' in params['price']
    
    def test_set_normalization_params(self, preprocessor):
        """Test setting normalization parameters."""
        params = {
            'price': {'mean': 11.0, 'std': 1.5},
            'sales_volume': {'mean': 150.0, 'std': 50.0}
        }
        
        preprocessor.set_normalization_params(params)
        
        retrieved_params = preprocessor.get_normalization_params()
        assert retrieved_params == params
    
    def test_original_data_not_modified(self, preprocessor, valid_data):
        """Test that original DataFrame is not modified."""
        original_columns = set(valid_data.columns)
        original_values = valid_data.copy()
        
        result = preprocessor.preprocess_features(valid_data, normalize=True)
        
        # Original data should not have new columns
        assert set(valid_data.columns) == original_columns
        # Original values should be unchanged
        pd.testing.assert_frame_equal(valid_data, original_values)
        # Result should have new columns
        assert 'price_normalized' in result.data.columns
    
    def test_multiple_errors_reported(self, preprocessor):
        """Test that multiple validation errors are reported."""
        data = pd.DataFrame({
            'price': [-10.0, np.nan, 11.0, np.inf],
            'is_holiday': ['yes', None, True, False]
        })
        
        result = preprocessor.preprocess_features(data)
        
        assert not result.is_valid
        # Should have errors for both price and holiday
        assert len(result.errors) >= 2
    
    def test_missing_columns_handled_gracefully(self, preprocessor):
        """Test that missing optional columns are handled gracefully."""
        data = pd.DataFrame({
            'sales_volume': [100.0, 150.0, 200.0]
        })
        
        result = preprocessor.preprocess_features(data, normalize=True)
        
        # Should succeed even without price and is_holiday
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_large_dataset_performance(self, preprocessor):
        """Test preprocessing on a larger dataset."""
        # Create 10,000 records
        data = pd.DataFrame({
            'price': np.random.uniform(5.0, 50.0, 10000),
            'sales_volume': np.random.uniform(50.0, 500.0, 10000),
            'is_holiday': np.random.choice([True, False], 10000)
        })
        
        result = preprocessor.preprocess_features(data, normalize=True)
        
        assert result.is_valid
        assert len(result.data) == 10000
        assert 'price_normalized' in result.data.columns
        assert 'sales_volume_normalized' in result.data.columns
    
    def test_edge_case_single_row(self, preprocessor):
        """Test preprocessing with single row."""
        data = pd.DataFrame({
            'price': [10.0],
            'is_holiday': [True]
        })
        
        result = preprocessor.preprocess_features(data, normalize=True)
        
        assert result.is_valid
        # Single row has zero std, should use std=1.0
        assert result.normalization_params['price']['std'] == 1.0
    
    def test_edge_case_all_same_values(self, preprocessor):
        """Test preprocessing when all values are the same."""
        data = pd.DataFrame({
            'price': [15.0] * 100,
            'sales_volume': [200.0] * 100,
            'is_holiday': [False] * 100
        })
        
        result = preprocessor.preprocess_features(data, normalize=True)
        
        assert result.is_valid
        # Constant features should have std=1.0
        assert result.normalization_params['price']['std'] == 1.0
        assert result.normalization_params['sales_volume']['std'] == 1.0
