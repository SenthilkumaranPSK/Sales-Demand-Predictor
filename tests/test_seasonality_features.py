"""
Unit tests for seasonality feature extraction.

Tests the extract_seasonality_features function to ensure correct extraction
of day_of_week, month, quarter, and season from timestamps.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.features.seasonality import extract_seasonality_features


class TestSeasonalityFeatureExtraction:
    """Test suite for seasonality feature extraction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various timestamps."""
        return pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-15',  # Winter, Monday
                '2024-03-20',  # Spring, Wednesday
                '2024-06-21',  # Summer, Friday
                '2024-09-22',  # Fall, Sunday
                '2024-12-25',  # Winter, Wednesday
            ]),
            'product_id': ['PROD_001'] * 5,
            'sales_volume': [100, 150, 200, 175, 250]
        })
    
    def test_extract_all_features(self, sample_data):
        """Test that all seasonality features are extracted."""
        result = extract_seasonality_features(sample_data)
        
        # Verify all expected columns are present
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'season' in result.columns
        
        # Verify original columns are preserved
        assert 'timestamp' in result.columns
        assert 'product_id' in result.columns
        assert 'sales_volume' in result.columns
    
    def test_day_of_week_extraction(self, sample_data):
        """Test day_of_week extraction (0=Monday, 6=Sunday)."""
        result = extract_seasonality_features(sample_data)
        
        # 2024-01-15 is Monday (0)
        assert result.loc[0, 'day_of_week'] == 0
        # 2024-03-20 is Wednesday (2)
        assert result.loc[1, 'day_of_week'] == 2
        # 2024-06-21 is Friday (4)
        assert result.loc[2, 'day_of_week'] == 4
        # 2024-09-22 is Sunday (6)
        assert result.loc[3, 'day_of_week'] == 6
        # 2024-12-25 is Wednesday (2)
        assert result.loc[4, 'day_of_week'] == 2
    
    def test_month_extraction(self, sample_data):
        """Test month extraction (1-12)."""
        result = extract_seasonality_features(sample_data)
        
        assert result.loc[0, 'month'] == 1   # January
        assert result.loc[1, 'month'] == 3   # March
        assert result.loc[2, 'month'] == 6   # June
        assert result.loc[3, 'month'] == 9   # September
        assert result.loc[4, 'month'] == 12  # December
    
    def test_quarter_extraction(self, sample_data):
        """Test quarter extraction (1-4)."""
        result = extract_seasonality_features(sample_data)
        
        assert result.loc[0, 'quarter'] == 1  # Q1 (Jan)
        assert result.loc[1, 'quarter'] == 1  # Q1 (Mar)
        assert result.loc[2, 'quarter'] == 2  # Q2 (Jun)
        assert result.loc[3, 'quarter'] == 3  # Q3 (Sep)
        assert result.loc[4, 'quarter'] == 4  # Q4 (Dec)
    
    def test_season_extraction(self, sample_data):
        """Test season extraction (spring, summer, fall, winter)."""
        result = extract_seasonality_features(sample_data)
        
        assert result.loc[0, 'season'] == 'winter'  # January
        assert result.loc[1, 'season'] == 'spring'  # March
        assert result.loc[2, 'season'] == 'summer'  # June
        assert result.loc[3, 'season'] == 'fall'    # September
        assert result.loc[4, 'season'] == 'winter'  # December
    
    def test_all_months_to_seasons(self):
        """Test season mapping for all 12 months."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([f'2024-{m:02d}-15' for m in range(1, 13)]),
            'product_id': ['PROD_001'] * 12
        })
        
        result = extract_seasonality_features(data)
        
        expected_seasons = [
            'winter',  # Jan
            'winter',  # Feb
            'spring',  # Mar
            'spring',  # Apr
            'spring',  # May
            'summer',  # Jun
            'summer',  # Jul
            'summer',  # Aug
            'fall',    # Sep
            'fall',    # Oct
            'fall',    # Nov
            'winter'   # Dec
        ]
        
        assert result['season'].tolist() == expected_seasons
    
    def test_custom_timestamp_column(self):
        """Test extraction with custom timestamp column name."""
        data = pd.DataFrame({
            'date': pd.to_datetime(['2024-06-15', '2024-12-15']),
            'product_id': ['PROD_001', 'PROD_002']
        })
        
        result = extract_seasonality_features(data, timestamp_column='date')
        
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert result.loc[0, 'season'] == 'summer'
        assert result.loc[1, 'season'] == 'winter'
    
    def test_missing_timestamp_column_raises_error(self):
        """Test that missing timestamp column raises ValueError."""
        data = pd.DataFrame({
            'product_id': ['PROD_001'],
            'sales_volume': [100]
        })
        
        with pytest.raises(ValueError, match="Timestamp column 'timestamp' not found"):
            extract_seasonality_features(data)
    
    def test_non_datetime_column_raises_error(self):
        """Test that non-datetime timestamp column raises ValueError."""
        data = pd.DataFrame({
            'timestamp': ['2024-01-15', '2024-06-15'],  # String, not datetime
            'product_id': ['PROD_001', 'PROD_002']
        })
        
        with pytest.raises(ValueError, match="must be datetime type"):
            extract_seasonality_features(data)
    
    def test_original_data_not_modified(self, sample_data):
        """Test that original DataFrame is not modified."""
        original_columns = set(sample_data.columns)
        
        result = extract_seasonality_features(sample_data)
        
        # Original data should not have new columns
        assert set(sample_data.columns) == original_columns
        # Result should have new columns
        assert 'day_of_week' in result.columns
        assert 'day_of_week' not in sample_data.columns
    
    def test_leap_year_handling(self):
        """Test correct handling of leap year dates."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-02-29']),  # Leap year
            'product_id': ['PROD_001']
        })
        
        result = extract_seasonality_features(data)
        
        assert result.loc[0, 'month'] == 2
        assert result.loc[0, 'quarter'] == 1
        assert result.loc[0, 'season'] == 'winter'
    
    def test_year_boundary_dates(self):
        """Test dates at year boundaries."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2023-12-31',  # End of year
                '2024-01-01',  # Start of year
            ]),
            'product_id': ['PROD_001', 'PROD_002']
        })
        
        result = extract_seasonality_features(data)
        
        # Both should be winter
        assert result.loc[0, 'season'] == 'winter'
        assert result.loc[1, 'season'] == 'winter'
        
        # Both should be in different quarters
        assert result.loc[0, 'quarter'] == 4  # Q4
        assert result.loc[1, 'quarter'] == 1  # Q1
    
    def test_large_dataset_performance(self):
        """Test extraction on a larger dataset."""
        # Create 10,000 records
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'product_id': ['PROD_001'] * 10000,
            'sales_volume': range(10000)
        })
        
        result = extract_seasonality_features(data)
        
        # Verify all features extracted
        assert len(result) == 10000
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'season' in result.columns
        
        # Verify value ranges
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        assert result['quarter'].min() >= 1
        assert result['quarter'].max() <= 4
        assert set(result['season'].unique()) == {'spring', 'summer', 'fall', 'winter'}
