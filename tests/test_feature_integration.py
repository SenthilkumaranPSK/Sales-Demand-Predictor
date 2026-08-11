"""
Integration tests for feature engineering pipeline.

Tests the integration between seasonality extraction and feature preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.seasonality import extract_seasonality_features
from src.features.preprocessing import FeaturePreprocessor


class TestFeatureIntegration:
    """Test suite for feature engineering pipeline integration."""
    
    @pytest.fixture
    def raw_data(self):
        """Create raw data with timestamps."""
        return pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-15',
                '2024-03-20',
                '2024-06-21',
                '2024-09-22',
                '2024-12-25',
            ]),
            'product_id': ['PROD_001'] * 5,
            'sales_volume': [100.0, 150.0, 200.0, 175.0, 250.0],
            'price': [10.0, 12.0, 15.0, 13.0, 18.0],
            'is_holiday': [False, False, False, False, True]
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create a feature preprocessor instance."""
        return FeaturePreprocessor()
    
    def test_full_feature_pipeline(self, raw_data, preprocessor):
        """Test complete feature engineering pipeline."""
        # Step 1: Extract seasonality features
        data_with_seasonality = extract_seasonality_features(raw_data)
        
        # Verify seasonality features are present
        assert 'day_of_week' in data_with_seasonality.columns
        assert 'month' in data_with_seasonality.columns
        assert 'quarter' in data_with_seasonality.columns
        assert 'season' in data_with_seasonality.columns
        
        # Step 2: Preprocess features (validate and normalize)
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        # Verify preprocessing succeeded
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Verify all features are present
        assert 'day_of_week' in result.data.columns
        assert 'month' in result.data.columns
        assert 'quarter' in result.data.columns
        assert 'season' in result.data.columns
        assert 'price_normalized' in result.data.columns
        assert 'sales_volume_normalized' in result.data.columns
        
        # Verify holiday is boolean
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
        
        # Verify normalization parameters are stored
        assert 'price' in result.normalization_params
        assert 'sales_volume' in result.normalization_params
    
    def test_pipeline_with_string_holidays(self, preprocessor):
        """Test pipeline with string holiday indicators."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-12-25']),
            'product_id': ['PROD_001', 'PROD_001'],
            'sales_volume': [100.0, 250.0],
            'price': [10.0, 18.0],
            'is_holiday': ['false', 'true']
        })
        
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(data)
        
        # Preprocess
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        assert result.is_valid
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
        assert result.data['is_holiday'].tolist() == [False, True]
    
    def test_pipeline_with_numeric_holidays(self, preprocessor):
        """Test pipeline with numeric holiday indicators."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-12-25']),
            'product_id': ['PROD_001', 'PROD_001'],
            'sales_volume': [100.0, 250.0],
            'price': [10.0, 18.0],
            'is_holiday': [0, 1]
        })
        
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(data)
        
        # Preprocess
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        assert result.is_valid
        assert pd.api.types.is_bool_dtype(result.data['is_holiday'])
        assert result.data['is_holiday'].tolist() == [False, True]
    
    def test_pipeline_train_test_split(self, raw_data, preprocessor):
        """Test pipeline with train/test split scenario."""
        # Extract seasonality for all data
        data_with_seasonality = extract_seasonality_features(raw_data)
        
        # Split into train and test
        train_data = data_with_seasonality.iloc[:3].copy()
        test_data = data_with_seasonality.iloc[3:].copy()
        
        # Fit preprocessing on training data
        train_result = preprocessor.preprocess_features(
            train_data,
            normalize=True,
            fit_normalization=True
        )
        
        assert train_result.is_valid
        
        # Transform test data using training parameters
        test_result = preprocessor.preprocess_features(
            test_data,
            normalize=True,
            fit_normalization=False
        )
        
        assert test_result.is_valid
        
        # Verify same normalization parameters are used
        assert train_result.normalization_params == test_result.normalization_params
    
    def test_pipeline_with_invalid_price(self, preprocessor):
        """Test pipeline rejects invalid price data."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-12-25']),
            'product_id': ['PROD_001', 'PROD_001'],
            'sales_volume': [100.0, 250.0],
            'price': [10.0, -5.0],  # Negative price
            'is_holiday': [False, True]
        })
        
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(data)
        
        # Preprocess should fail
        result = preprocessor.preprocess_features(data_with_seasonality)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('negative' in error.lower() for error in result.errors)
    
    def test_pipeline_with_invalid_holiday(self, preprocessor):
        """Test pipeline rejects invalid holiday data."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-12-25']),
            'product_id': ['PROD_001', 'PROD_001'],
            'sales_volume': [100.0, 250.0],
            'price': [10.0, 18.0],
            'is_holiday': ['yes', 'no']  # Invalid values
        })
        
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(data)
        
        # Preprocess should fail
        result = preprocessor.preprocess_features(data_with_seasonality)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('invalid values' in error.lower() for error in result.errors)
    
    def test_pipeline_preserves_seasonality_features(self, raw_data, preprocessor):
        """Test that preprocessing preserves seasonality feature values."""
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(raw_data)
        
        # Store original seasonality values
        original_day_of_week = data_with_seasonality['day_of_week'].copy()
        original_month = data_with_seasonality['month'].copy()
        original_quarter = data_with_seasonality['quarter'].copy()
        original_season = data_with_seasonality['season'].copy()
        
        # Preprocess
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        # Verify seasonality features are unchanged
        pd.testing.assert_series_equal(
            result.data['day_of_week'],
            original_day_of_week,
            check_names=False
        )
        pd.testing.assert_series_equal(
            result.data['month'],
            original_month,
            check_names=False
        )
        pd.testing.assert_series_equal(
            result.data['quarter'],
            original_quarter,
            check_names=False
        )
        pd.testing.assert_series_equal(
            result.data['season'],
            original_season,
            check_names=False
        )
    
    def test_pipeline_large_dataset(self, preprocessor):
        """Test pipeline with larger dataset."""
        # Create 1000 records
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'product_id': ['PROD_001'] * 1000,
            'sales_volume': np.random.uniform(50.0, 500.0, 1000),
            'price': np.random.uniform(5.0, 50.0, 1000),
            'is_holiday': np.random.choice([True, False], 1000)
        })
        
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(data)
        
        # Preprocess
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        assert result.is_valid
        assert len(result.data) == 1000
        assert 'price_normalized' in result.data.columns
        assert 'sales_volume_normalized' in result.data.columns
        
        # Verify normalized features have mean ~0 and std ~1
        assert abs(result.data['price_normalized'].mean()) < 0.1
        assert abs(result.data['price_normalized'].std() - 1.0) < 0.1
    
    def test_pipeline_denormalization(self, raw_data, preprocessor):
        """Test denormalization after full pipeline."""
        # Extract seasonality
        data_with_seasonality = extract_seasonality_features(raw_data)
        
        # Preprocess with normalization
        result = preprocessor.preprocess_features(data_with_seasonality, normalize=True)
        
        # Get normalized prices
        normalized_prices = result.data['price_normalized'].values
        
        # Denormalize
        denormalized_prices = preprocessor.denormalize_feature(normalized_prices, 'price')
        
        # Verify denormalized values match original
        original_prices = raw_data['price'].values
        assert np.allclose(denormalized_prices, original_prices)
